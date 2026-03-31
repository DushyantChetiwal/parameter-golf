"""
Annealed-Muon 1.58-bit model utilizing Grouped Parameter Tying
Compliant with the 10-minute / 16MB Parameter Golf Constraints.
Featuring Dynamic Hardware Self-Calibration and Dual-Clock Annealing.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    def __init__(self):
        # -----------------------------
        # 1. BASE CONFIGURATION (A100 Defaults)
        # -----------------------------
        self.data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
        self.run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
        self.seed = int(os.environ.get("SEED", 1337))

        self.val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))
        self.muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 100))

        self.iterations = int(os.environ.get("ITERATIONS", 3000))
        self.warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 0))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", 5)) 
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

        self.vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
        self.num_layers = int(os.environ.get("NUM_LAYERS", 12)) 
        self.num_unique_blocks = int(os.environ.get("NUM_UNIQUE_BLOCKS", 12))
        self.model_dim = int(os.environ.get("MODEL_DIM", 1536))
        self.num_heads = int(os.environ.get("NUM_HEADS", 16))
        self.mlp_mult = int(os.environ.get("MLP_MULT", 4))
        
        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
        self.tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
        self.rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

        self.embed_lr = float(os.environ.get("EMBED_LR", 0.05))
        self.head_lr = float(os.environ.get("HEAD_LR", 0.008))
        self.tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
        
        self.matrix_lr = float(os.environ.get("MATRIX_LR", 0.4))
        self.scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
        
        self.muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.90)) 
        self.muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
        self.muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
        
        self.beta1 = float(os.environ.get("BETA1", 0.9))
        self.beta2 = float(os.environ.get("BETA2", 0.95))
        self.adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
        self.latent_clip_scale = float(os.environ.get("LATENT_CLIP_SCALE", 3.0))

        # -----------------------------
        # 2. 🚀 THE HARDWARE AUTO-SCALER
        # -----------------------------
        if torch.cuda.is_available():
            gpu_capability = torch.cuda.get_device_capability(0)[0] # 8 = Ampere, 9 = Hopper
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            
            # Trigger Criteria: Is this an H100 (Capability 9+) OR a massive 8x GPU cluster?
            if gpu_capability >= 9 or world_size >= 8:
                
                # Quadruple the batch sizes to stop SM starvation
                # (We check `not in os.environ` so you can still manually override if you want)
                if "TRAIN_BATCH_TOKENS" not in os.environ:
                    self.train_batch_tokens = 1_048_576
                if "VAL_BATCH_SIZE" not in os.environ:
                    self.val_batch_size = 1_048_576
                    
                # Double the learning rates to obey the Batch Size Scaling Law
                if "MATRIX_LR" not in os.environ:
                    self.matrix_lr = 0.8
                    self.scalar_lr = 0.08
                    self.embed_lr = 0.10
                    self.tied_embed_lr = 0.10
                    
                # Squeeze more precision out of Muon since H100 compute is essentially free
                if "MUON_BACKEND_STEPS" not in os.environ:
                    self.muon_backend_steps = 7

# -----------------------------
# MUON OPTIMIZER & EVALUATION
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            updates_flat = torch.zeros(sum(int(p.numel()) for p in params), device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                p.add_(updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype), alpha=-lr)
                curr += p.numel()
        return loss

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes_np, has_leading_space_np, is_boundary_token_np = np.zeros((table_size,), dtype=np.int16), np.zeros((table_size,), dtype=np.bool_), np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device), torch.tensor(has_leading_space_np, dtype=torch.bool, device=device), torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([torch.from_numpy(np.fromfile(file, dtype="<u2", count=int(np.fromfile(file, dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False)) for file in files]).contiguous()
    return tokens[: ((tokens.numel() - 1) // seq_len) * seq_len + 1]

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device, grad_accum_steps: int, val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor) -> tuple[float, float]:
    local_batch_seqs = (args.val_batch_size // (world_size * grad_accum_steps)) // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start, seq_end = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    val_loss_sum, val_token_count, val_byte_count = torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    tau_eval, step_eval = torch.tensor([0.01], device=device, dtype=torch.float32), torch.tensor([1.0], device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            local = val_tokens[batch_seq_start * args.train_seq_len : batch_seq_end * args.train_seq_len + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                val_loss_sum += model(x, y, tau=tau_eval, step_fraction=step_eval).detach().to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            token_bytes = base_bytes_lut[y.reshape(-1)].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[y.reshape(-1)] & ~is_boundary_token_lut[x.reshape(-1)]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM); dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM); dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return float((val_loss_sum / val_token_count).item()), float((val_loss_sum / val_token_count).item() / math.log(2.0) * (val_token_count.item() / val_byte_count.item()))

def pack_base3_uint8(w_latent: Tensor) -> tuple[Tensor, int, int, int]:
    out_features, in_features = w_latent.shape
    d = (w_latent.clamp(-1.0, 1.0).round() + 1.0).to(torch.uint8).flatten() 
    pad_len = (5 - (d.numel() % 5)) % 5
    if pad_len != 0: d = F.pad(d, (0, pad_len), value=1)
    return (d.view(-1, 5).to(torch.int32) * torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=d.device)).sum(dim=1).to(torch.uint8), out_features, in_features, pad_len

def unpack_base3_uint8(v_packed: Tensor, out_features: int, in_features: int, pad_len: int) -> Tensor:
    d = ((v_packed.to(torch.int32).unsqueeze(1) // torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=v_packed.device)) % 3).flatten()
    if pad_len > 0: d = d[:-pad_len]
    return (d.to(torch.bfloat16) - 1.0).view(out_features, in_features)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.files, self.rank, self.world_size, self.device, self.file_idx, self.pos = [Path(p) for p in sorted(glob.glob(pattern))], rank, world_size, device, 0, 0
        self.tokens = torch.from_numpy(np.fromfile(self.files[0], dtype="<u2", count=int(np.fromfile(self.files[0], dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False))

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        per_rank_span = (global_tokens // (self.world_size * grad_accum_steps)) + 1
        n = per_rank_span * self.world_size
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = torch.from_numpy(np.fromfile(self.files[self.file_idx], dtype="<u2", count=int(np.fromfile(self.files[self.file_idx], dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False))
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        local = (chunks[0] if len(chunks) == 1 else torch.cat(chunks))[self.rank * per_rank_span : (self.rank + 1) * per_rank_span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True), local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class AnnealedBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight_latent = nn.Parameter(torch.empty(out_features, in_features))
        
        # FIX 1: Widen the initialization so weights aren't born inside the 'Zero' deadzone!
        nn.init.normal_(self.weight_latent, mean=0.0, std=0.2) 
        
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.register_parameter('bias', None)
        self._zero_init = False

    def forward(self, x: Tensor, tau: Tensor, step_fraction: Tensor) -> Tensor:
        # FIX 2: The mathematically pure, graph-friendly Linear STE. 
        # No tanh dead-zones. No clamp explosions.
        
        # 1. The pure ternary target
        w_quant = self.weight_latent.round().clamp(-1.0, 1.0)
        
        # 2. Smooth linear interpolation from soft-weights to hard-ternary
        W_active = self.weight_latent + (step_fraction * (w_quant - self.weight_latent)).detach()
        
        return F.linear(x, W_active.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self.cos_cached[:, :, :seq_len, :].to(dtype), self.sin_cached[:, :, :seq_len, :].to(dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.c_q = AnnealedBitLinear(dim, dim, bias=False)
        self.c_k = AnnealedBitLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = AnnealedBitLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = AnnealedBitLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, tau: Tensor, step_fraction: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, tau, step_fraction).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x, tau, step_fraction).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x, tau, step_fraction).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, q.dtype)
        q = apply_rotary_emb(q, cos, sin) * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        k = apply_rotary_emb(k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim), tau, step_fraction)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = AnnealedBitLinear(dim, mlp_mult * dim, bias=False)
        self.proj = AnnealedBitLinear(mlp_mult * dim, dim, bias=False)

    def forward(self, x: Tensor, tau: Tensor, step_fraction: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x, tau, step_fraction)).square(), tau, step_fraction)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, tau: Tensor, step_fraction: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x), tau, step_fraction)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x), tau, step_fraction)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_unique_blocks: int, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.tie_embeddings, self.logit_softcap = tie_embeddings, logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
            
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, CastedLinear, AnnealedBitLinear)):
                w = getattr(module, 'weight_latent', getattr(module, 'weight', None))
                if w is not None:
                    if getattr(module, "_zero_init", False): 
                        nn.init.zeros_(w)
                    elif isinstance(module, AnnealedBitLinear):
                        # FIX: Protect the 1.58-bit layers! Force the wide distribution
                        # so they are born outside the ternary dead-zones.
                        nn.init.normal_(w, mean=0.0, std=0.6)
                    else:
                        # Only apply the standard GPT orthogonal scaling to standard linear layers (like lm_head)
                        nn.init.orthogonal_(w)
                        if ".proj" in name:
                            with torch.no_grad(): w.mul_(1.0 / math.sqrt(2 * num_layers))

        # --- NATIVE PARAMETER TYING (GROUPED) ---
        for i in range(num_layers):
            leader_index = i % num_unique_blocks 
            if i == leader_index: continue 
                
            block, leader = self.blocks[i], self.blocks[leader_index]
            block.attn.c_q.weight_latent = leader.attn.c_q.weight_latent
            block.attn.c_k.weight_latent = leader.attn.c_k.weight_latent
            block.attn.c_v.weight_latent = leader.attn.c_v.weight_latent
            block.attn.proj.weight_latent = leader.attn.proj.weight_latent
            block.mlp.fc.weight_latent = leader.mlp.fc.weight_latent
            block.mlp.proj.weight_latent = leader.mlp.proj.weight_latent
            block.attn.q_gain = leader.attn.q_gain

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None, tau: Tensor | None = None, step_fraction: Tensor | None = None) -> Tensor:
        if tau is None: tau = torch.tensor([0.01], device=input_ids.device, dtype=torch.float32)
        if step_fraction is None: step_fraction = torch.tensor([1.0], device=input_ids.device, dtype=torch.float32)

        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(-1),))
        for block in self.blocks: x = block(x, tau, step_fraction)
        
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.logit_softcap * torch.tanh((F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)) / self.logit_softcap)
        
        if target_ids is not None: return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        return logits


# -----------------------------
# UPGRADED DIAGNOSTIC SNIFFER
# -----------------------------
@torch.no_grad()
def measure_quantization_health(model: nn.Module) -> tuple[float, float]:
    total_params = 0
    stuck_at_zero = 0
    active_edges = 0
    processed_ids = set()
    
    for module in model.modules():
        if isinstance(module, AnnealedBitLinear):
            w = module.weight_latent
            wid = id(w)
            if wid not in processed_ids:
                processed_ids.add(wid)
                w_discrete = w.clamp(-1.0, 1.0).round()
                
                # Count weights resting near 0
                stuck_at_zero += (torch.abs(w) < 0.15).sum().item()
                # Count weights resting near -1 or 1
                active_edges += ((torch.abs(w - w_discrete) < 0.15) & (torch.abs(w_discrete) > 0.5)).sum().item()
                
                total_params += w.numel()
                
    zero_pct = (stuck_at_zero / max(total_params, 1)) * 100.0
    edge_pct = (active_edges / max(total_params, 1)) * 100.0
    return zero_pct, edge_pct


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    absolute_t0 = time.perf_counter() # ABSOLUTE CLOCK START
    
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank, world_size, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    
    if 32 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 32")
    grad_accum_steps = 32 // world_size
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    if master_process: os.makedirs("logs", exist_ok=True)
    def log0(msg: str):
        if master_process:
            print(msg)
            if logfile: open(logfile, "a", encoding="utf-8").write(msg + "\n")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = GPT(args.vocab_size, args.num_layers, args.num_unique_blocks, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std, args.logit_softcap, args.rope_base, args.qk_gain_init).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
        
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if (param.ndim < 2 or "scale" in name or "norm" in name) and param.dtype != torch.float32:
                param.data = param.data.float()
                
    # --- COMPILER BYPASSED FOR EAGER MODE SPEED ---
    # for i in range(len(base_model.blocks)):
    #     base_model.blocks[i] = torch.compile(base_model.blocks[i], fullgraph=True, dynamic=False)
    
    # 1. Compile the base_model BEFORE wrapping in DDP (This fixes the 3-minute submod fracturing)
    base_model = torch.compile(base_model, fullgraph=False)
    
    # 2. Wrap in DDP
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    matrix_params, scalar_params = [], []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or "lm_head" in name: continue
        if p.ndim == 2 and "weight_latent" in name: matrix_params.append(p)
        else: scalar_params.append(p)
            
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizers = [
        torch.optim.AdamW([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True),
        Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps),
        torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    ]
    for group in optimizers[1].param_groups: group["base_lr"] = args.matrix_lr

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # --- THE COMPREHENSIVE SELF-CALIBRATION BRAIN ---
    args.warmup_steps = 5 
    
    if args.warmup_steps > 0:
        model.train()
        
        # 1. THE PRIMER STEP (This absorbs the 74-second compile tax)
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                (model(x, y, tau=torch.tensor([1.0], device=device), step_fraction=torch.tensor([0.0], device=device)) * (1.0/grad_accum_steps)).backward()
        for opt in optimizers: opt.step()
        torch.cuda.synchronize() # Wait for the compiler to finish writing the GPU kernels
        
        # 2. THE TRUE CALIBRATION RUN (Now we start the stopwatch)
        t_warmup_start = time.perf_counter()
        for _ in range(args.warmup_steps):
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    (model(x, y, tau=torch.tensor([1.0], device=device), step_fraction=torch.tensor([0.0], device=device)) * (1.0/grad_accum_steps)).backward()
            for opt in optimizers: opt.step()
        torch.cuda.synchronize()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    warmup_time_ms = 1000.0 * (time.perf_counter() - t_warmup_start)
    avg_step_ms = warmup_time_ms / max(1, args.warmup_steps)
    time_elapsed_ms = 1000.0 * (time.perf_counter() - absolute_t0)
    
    # ----------------------------------------------------
    # NEW: Synchronize the calibration timings from Rank 0!
    # ----------------------------------------------------
    if distributed:
        sync_metrics = torch.tensor([time_elapsed_ms, avg_step_ms], dtype=torch.float32, device=device)
        dist.broadcast(sync_metrics, src=0)
        time_elapsed_ms, avg_step_ms = sync_metrics.tolist()
        
    # 1. Dynamic Safety Buffer
    safety_buffer_ms = (avg_step_ms * 2.5) + 5000.0 
    time_left_ms = (args.max_wallclock_seconds * 1000.0) - time_elapsed_ms - safety_buffer_ms
    
    projected_main_steps = max(1, int(time_left_ms / avg_step_ms))
    # 2. Dynamic Muon Momentum
    args.muon_momentum_warmup_steps = max(2, int(projected_main_steps * 0.10))
    
    # 3. Dynamic Validation
    args.val_loss_every = max(1, projected_main_steps // 3)
    
    # 4. Dynamic Logging
    args.train_log_every = max(1, projected_main_steps // 20)
    
    log0("\n" + "="*50)
    log0(f"🚀 HARDWARE AUTO-CALIBRATION COMPLETE")
    log0(f"   - Hardware Speed:    {avg_step_ms:.0f} ms/step")
    log0(f"   - Projected Steps:   {projected_main_steps} steps remaining")
    log0(f"   - Muon Warmup Phase: {args.muon_momentum_warmup_steps} steps")
    log0(f"   - Validation Freq:   Every {args.val_loss_every} steps")
    log0(f"   - Sniffer Log Freq:  Every {args.train_log_every} steps")
    log0(f"   - I/O Safety Buffer: {safety_buffer_ms/1000.0:.1f} seconds")
    log0("="*50 + "\n")
    
    # --- CLOCK SEPARATION ---
    loop_max_ms = time_left_ms 
    
    # 1. Initialize our accumulators
    training_time_ms = 0.0 
    t0 = time.perf_counter() 
    
    step = 0
    stop_after_step = None
    
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            
            # 2. PAUSE THE CLOCK: Add the time spent training so far to the accumulator
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
            
            # 3. RESTART THE CLOCK: Reset t0 after validation finishes
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step: break

        # --- TIME-DOMAIN ANNEALING (Perfectly Scaled to Remaining Time) ---
        # 4. Use the accumulated training time to drive the math
        loop_elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        fraction = min(loop_elapsed_ms / loop_max_ms, 1.0)
        cos_fraction = 0.5 * (1 + math.cos(math.pi * fraction))
        
        tau_val = 0.01 + (1.0 - 0.01) * cos_fraction
        tau_t, frac_t = torch.tensor([tau_val], device=device, dtype=torch.float32), torch.tensor([fraction], device=device, dtype=torch.float32)
        scale = cos_fraction 
        # --------------------------------------------------

        for opt in optimizers: opt.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): 
                loss = model(x, y, tau=tau_t, step_fraction=frac_t)
            train_loss += loss.detach()
            (loss * (1.0/grad_accum_steps)).backward()
        train_loss /= grad_accum_steps

        frac_muon = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizers[1].param_groups: group["momentum"] = (1 - frac_muon) * args.muon_momentum_warmup_start + frac_muon * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        for opt in optimizers: opt.step()
            
        with torch.no_grad():
            bound = 1.0 + args.latent_clip_scale * tau_val
            for module in base_model.modules():
                if isinstance(module, AnnealedBitLinear): module.weight_latent.clamp_(-bound, bound)

        step += 1
        
        # --- THE ASSASSIN CLOCK (Kill Switch) ---
        total_absolute_ms = 1000.0 * (time.perf_counter() - absolute_t0)
        
        reached_cap = max_wallclock_ms is not None and total_absolute_ms >= (max_wallclock_ms - safety_buffer_ms)
        if distributed and max_wallclock_ms is not None:
            cap_tensor = torch.tensor([1 if reached_cap else 0], dtype=torch.int32, device=device)
            dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = cap_tensor.item() > 0

        if stop_after_step is None and reached_cap:
            log0(f"stopping_early: cap reached at step {step} (Absolute time: {total_absolute_ms/1000.0:.1f}s)")
            stop_after_step = step 
        
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            zero_pct, edge_pct = measure_quantization_health(base_model)
            log0(f"step:{step} train_loss:{train_loss.item():.4f} "
                 f"step_avg:{loop_elapsed_ms/max(step,1):.0f}ms "
                 f"[Zero: {zero_pct:.1f}% | Edges: {edge_pct:.1f}%]")

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        packed_state_dict, packed_ids = {}, set()
        for name, module in base_model.named_modules():
            if isinstance(module, AnnealedBitLinear):
                wid = id(module.weight_latent)
                if wid not in packed_ids:
                    packed_ids.add(wid)
                    v_packed, out_features, in_features, pad_len = pack_base3_uint8(module.weight_latent.detach())
                    packed_state_dict[f"{name}.v_packed"] = v_packed.cpu()
                    packed_state_dict[f"{name}.shape_info"] = torch.tensor([out_features, in_features, pad_len], dtype=torch.int32).cpu()
                
        for name, param in base_model.state_dict().items():
            if "weight_latent" not in name: packed_state_dict[name] = param.detach().cpu()

        torch.save(packed_state_dict, "final_model.pt")
        quant_buf = io.BytesIO()
        torch.save(packed_state_dict, quant_buf)
        open("final_model.int8.ptz", "wb").write(zlib.compress(quant_buf.getvalue(), level=9))
        log0(f"Serialized model packed+zlib: {os.path.getsize('final_model.int8.ptz')} bytes")

    if distributed: dist.barrier()
        
    packed_state_dict = torch.load(io.BytesIO(zlib.decompress(open("final_model.int8.ptz", "rb").read())), map_location="cpu")
    dequant_state_dict = {}
    for name, param in packed_state_dict.items():
        if name.endswith(".v_packed"):
            prefix = name[:-len(".v_packed")]
            out_features, in_features, pad_len = packed_state_dict[f"{prefix}.shape_info"].tolist()
            dequant_state_dict[f"{prefix}.weight_latent"] = unpack_base3_uint8(param, out_features, in_features, pad_len)
        elif not name.endswith(".shape_info"): dequant_state_dict[name] = param

    base_model.load_state_dict(dequant_state_dict, strict=False)
            
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

if __name__ == "__main__":
    main()