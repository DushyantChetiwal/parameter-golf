"""
Annealed-Muon 1.58-bit model utilizing a depth recurrent Core Brain
Compliant with the 10-minute / 16MB Parameter Golf Constraints.
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
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 12000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 0))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12)) 
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 2048))
    num_heads = int(os.environ.get("NUM_HEADS", 16))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    
    latent_clip_scale = float(os.environ.get("LATENT_CLIP_SCALE", 3.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION 
# -----------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device, grad_accum_steps: int, val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE must provide at least one sequence per rank")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    tau_eval = torch.tensor([0.01], device=device, dtype=torch.float32)
    step_eval = torch.tensor([1.0], device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, tau=tau_eval, step_fraction=step_eval).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# BASE-3 PACKING (1.58 BIT)
# -----------------------------

def pack_base3_uint8(w_latent: Tensor) -> tuple[Tensor, int, int, int]:
    out_features, in_features = w_latent.shape
    w_discrete = w_latent.clamp(-1.0, 1.0).round()
    d = (w_discrete + 1.0).to(torch.uint8).flatten() 
    remainder = d.numel() % 5
    if remainder != 0:
        pad_len = 5 - remainder
        d = F.pad(d, (0, pad_len), value=1)
    else:
        pad_len = 0
        
    d_view = d.view(-1, 5).to(torch.int32)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=d.device)
    v_packed = (d_view * powers).sum(dim=1).to(torch.uint8)
    return v_packed, out_features, in_features, pad_len

def unpack_base3_uint8(v_packed: Tensor, out_features: int, in_features: int, pad_len: int) -> Tensor:
    v_int = v_packed.to(torch.int32).unsqueeze(1)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=v_packed.device)
    d = (v_int // powers) % 3 
    d_flat = d.flatten()
    if pad_len > 0:
        d_flat = d_flat[:-pad_len]
    w_discrete = d_flat.to(torch.bfloat16) - 1.0
    return w_discrete.view(out_features, in_features)

# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or "gamma" in name or "beta" in name) and param.dtype != torch.float32:
                param.data = param.data.float()

# --- OPTIMIZATION 1: Standalone Compiled Math to prevent dynamic loop tracing ---
@torch.compile(fullgraph=True, dynamic=False)
def compute_active_weight(weight_latent: Tensor, tau: Tensor, step_fraction: Tensor) -> Tensor:
    tau_safe = tau.clamp(min=1e-4)
    W_active_tanh = 0.5 * (
        torch.tanh((weight_latent + 0.5) / tau_safe) +
        torch.tanh((weight_latent - 0.5) / tau_safe)
    )
    w_discrete = weight_latent.clamp(-1.0, 1.0).round()
    W_active_snap = (w_discrete - weight_latent).detach() + weight_latent
    return torch.where(step_fraction > 0.95, W_active_snap, W_active_tanh)

class AnnealedBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_latent = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight_latent, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self._zero_init = False

    def get_active_weight(self, tau: Tensor, step_fraction: Tensor) -> Tensor:
        return compute_active_weight(self.weight_latent, tau, step_fraction)

# --- OPTIMIZATION 2: Pre-Cached Rotary to prevent 'is None' Graph Breaks ---
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self.cos_cached[:, :, :seq_len, :].to(dtype), self.sin_cached[:, :, :seq_len, :].to(dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        
        self.c_q = AnnealedBitLinear(dim, dim, bias=False)
        self.c_k = AnnealedBitLinear(dim, kv_dim, bias=False)
        self.c_v = AnnealedBitLinear(dim, kv_dim, bias=False)
        self.proj = AnnealedBitLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, w_proj: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, w_q.to(x.dtype), self.c_q.bias).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, w_k.to(x.dtype), self.c_k.bias).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, w_v.to(x.dtype), self.c_v.bias).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads)
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return F.linear(y, w_proj.to(x.dtype), self.proj.bias)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = AnnealedBitLinear(dim, hidden, bias=False)
        self.proj = AnnealedBitLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, w_fc: Tensor, w_proj: Tensor) -> Tensor:
        x = torch.relu(F.linear(x, w_fc.to(x.dtype), self.fc.bias))
        return F.linear(x.square(), w_proj.to(x.dtype), self.proj.bias)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, w_attn_proj: Tensor, w_mlp_fc: Tensor, w_mlp_proj: Tensor) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), w_q, w_k, w_v, w_attn_proj)
        delta_attn = self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x_mid = x + delta_attn
        delta_mlp = self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_mid), w_mlp_fc, w_mlp_proj)
        return delta_attn + delta_mlp

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        
        self.num_loops = num_layers
        self.sub_ln = RMSNorm()
        
        self.core_brain = Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
        
        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(model_dim)) for _ in range(self.num_loops)])
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(model_dim)) for _ in range(self.num_loops)])

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
            
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, CastedLinear, AnnealedBitLinear)):
                w = getattr(module, 'weight_latent', getattr(module, 'weight', None))
                if w is not None:
                    if getattr(module, "_zero_init", False):
                        nn.init.zeros_(w)
                    else:
                        nn.init.orthogonal_(w)
                        if ".proj." in name or name.endswith(".proj"):
                            with torch.no_grad():
                                w.mul_(1.0 / math.sqrt(2 * self.num_loops))

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None, tau: Tensor | None = None, step_fraction: Tensor | None = None) -> Tensor:
        if tau is None:
            tau = torch.tensor([0.01], device=input_ids.device, dtype=torch.float32)
        if step_fraction is None:
            step_fraction = torch.tensor([1.0], device=input_ids.device, dtype=torch.float32)

        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        
        cb = self.core_brain
        w_q = cb.attn.c_q.get_active_weight(tau, step_fraction)
        w_k = cb.attn.c_k.get_active_weight(tau, step_fraction)
        w_v = cb.attn.c_v.get_active_weight(tau, step_fraction)
        w_attn_proj = cb.attn.proj.get_active_weight(tau, step_fraction)
        w_mlp_fc = cb.mlp.fc.get_active_weight(tau, step_fraction)
        w_mlp_proj = cb.mlp.proj.get_active_weight(tau, step_fraction)

        # Loop evaluates strictly in eager Python.
        for l in range(self.num_loops):
            normalized_x = self.sub_ln(x)
            # The core_brain itself will be a compiled object.
            core_out = cb(normalized_x, w_q, w_k, w_v, w_attn_proj, w_mlp_fc, w_mlp_proj)
            x = x + core_out * self.gamma[l].to(x.dtype) + self.beta[l].to(x.dtype)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
            
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        
        if target_ids is not None:
            targets = target_ids.reshape(-1)
            return F.cross_entropy(logits.float(), targets, reduction="mean")
        return logits

# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 32 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 32 so grad_accum_steps stays integral")
    grad_accum_steps = 32 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
        
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    
    # --- OPTIMIZATION 3: Targeted Compilation to prevent Loop Unrolling Explosion ---
    # We strictly target torch.compile at the single CoreBrain block.
    # The 12-loop executes in Python, preventing Dynamo from generating an 840M param graph.
    base_model.core_brain = torch.compile(base_model.core_brain, fullgraph=True, dynamic=False)
    
    # DDP perfectly wraps the eager model, handling the compiled sub-module natively.
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    matrix_params = []
    scalar_params = []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or "lm_head" in name:
            continue
        if p.ndim == 2 and "weight_latent" in name:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
            
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_weight_decay if hasattr(args, 'adam_weight_decay') else 0.0, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_weight_decay if hasattr(args, 'adam_weight_decay') else 0.0, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y, tau=torch.tensor([1.0], device=device), step_fraction=torch.tensor([0.0], device=device))
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        fraction = min(step / args.iterations, 1.0)
        cos_fraction = 0.5 * (1 + math.cos(math.pi * fraction))
        tau_val = 0.01 + (1.0 - 0.01) * cos_fraction
        
        tau_tensor = torch.tensor([tau_val], device=device, dtype=torch.float32)
        fraction_tensor = torch.tensor([fraction], device=device, dtype=torch.float32)
        
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, tau=tau_tensor, step_fraction=fraction_tensor)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac_muon = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac_muon) * args.muon_momentum_warmup_start + frac_muon * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            
        for opt in optimizers:
            opt.step()
            
        # Proximal Clipping natively in eager python (outside compilation graph)
        with torch.no_grad():
            bound = 1.0 + args.latent_clip_scale * tau_val
            for module in base_model.modules():
                if isinstance(module, AnnealedBitLinear):
                    module.weight_latent.clamp_(-bound, bound)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_tensor = torch.tensor([1 if reached_cap else 0], dtype=torch.int32, device=device)
            dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = cap_tensor.item() > 0

        if stop_after_step is None and reached_cap:
            log0(f"stopping_early: max_wallclock_ms cap reached at step {step}")
            stop_after_step = step 
        
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        packed_state_dict = {}
        for name, module in base_model.named_modules():
            if isinstance(module, AnnealedBitLinear):
                v_packed, out_features, in_features, pad_len = pack_base3_uint8(module.weight_latent.detach())
                packed_state_dict[f"{name}.v_packed"] = v_packed.cpu()
                packed_state_dict[f"{name}.shape_info"] = torch.tensor([out_features, in_features, pad_len], dtype=torch.int32).cpu()
                
        for name, param in base_model.state_dict().items():
            if "weight_latent" not in name and "tau" not in name and "step_fraction" not in name:
                packed_state_dict[name] = param.detach().cpu()

        torch.save(packed_state_dict, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        
        quant_buf = io.BytesIO()
        torch.save(packed_state_dict, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
            
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model packed+zlib: {quant_file_bytes} bytes (raw_dict:{model_bytes})")
        log0(f"Total submission size packed+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
        
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    packed_state_dict = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    
    dequant_state_dict = {}
    for name, param in packed_state_dict.items():
        if name.endswith(".v_packed"):
            prefix = name[:-len(".v_packed")]
            shape_info = packed_state_dict[f"{prefix}.shape_info"]
            out_features, in_features, pad_len = shape_info.tolist()
            w_discrete = unpack_base3_uint8(param, out_features, in_features, pad_len)
            dequant_state_dict[f"{prefix}.weight_latent"] = w_discrete
        elif name.endswith(".shape_info"):
            continue
        else:
            dequant_state_dict[name] = param

    base_model.load_state_dict(dequant_state_dict, strict=False)
            
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()