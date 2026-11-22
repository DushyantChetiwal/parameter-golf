# RD-$E_8$ Looped Distillation (Zero-Codebook Spherical Shell-$E_8$ Lattice Codec)

**Canonical Full-Validation Result (`47,853,343` CaseOps tokens, `151,080,645` bytes, `<= 16,000,000 B` total):**
- **Score-First LoRA TTT `val_bpb`:** **`1.099024`** (`val_loss = 2.405078`, `543.7s <= 600s` on `8×H100`) — beats the `#1` Unlimited-Compute 1-bit Parameter Golf entry (`1.1239`) by **`-0.0249` bpb**.
- **Zero-TTT Sliding Window (`stride=256`) `val_bpb`:** **`1.107200`** (`50.0s` on `8×H100`) — beats `1.1239` by **`-0.0167` bpb** without TTT.
- **Total Artifact Footprint (model + uncompressed source code):** **`15,828,534 B`** (`15,753,857 B` raw `RDE8` binary + `74,677 B` code — `171 KB` strictly under `16,000,000 B`).

---

## 1. Method

1. **Zero-Codebook Spherical Shell-$E_8$ Lattice Codec (`train_gpt.py`):**
   - Applies block-256 orthogonal Fast Walsh–Hadamard Transform (`fwht_block`) so weight blocks follow an isotropic Gaussian distribution.
   - Snaps every 8D weight block onto the Gosset lattice $E_8 = D_8 \cup (D_8 + \tfrac{1}{2})$ via Conway–Sloane nearest-point decoding (`snap_e8`).
   - Losslessly encodes each 8D lattice point by its spherical shell $m = \|q\|^2 / 2$ (entropy-coded via Brotli-11) plus its exact **DP enumerative sphere index** inside shell $m$ (`0` bytes of codebook overhead, `~163ms` encode / `~99ms` decode per 1M weights, bit-exact roundtrip).
   - At eval/TTT time, symmetric orthogonal FWHT folds into the unpacked weight matrix (`freeze_for_eval()`), running at native cuBLAS BF16 GEMM speed.

2. **Rate-Distortion $E_8$ Distillation (`--stage rd_e8_kd`):**
   - Trains a `53.2M` unique / `75.5M` effective looped Transformer (`8` unique blocks, blocks `[2..5]` looped twice → `12` effective layers, `d=768`, GQA `12/4`, `4× ReLU^2` MLP, U-Net skips, SmearGate, bottlenecked `8192×320` embeddings) in BF16 with Muon + AdamW.
   - Initializes `RDE8Linear` layers via zero-shot Hadamard + $E_8$ snapping (`0.8s`), then minimizes $(1 - \alpha)\,\text{CE} + \alpha\,\text{KL} + \lambda \cdot \widehat{R}(W_{\text{rot}}, \delta)$ where $\widehat{R}$ is a smooth differentiable proxy for Shell-$E_8$ bits and $\lambda$ adapts automatically to the exact byte target.

---

## 2. Controlled `1×H100` Ablation: Beating the Dense BF16 Teacher Inside `14.80 MB`

All arms ran on the identical `53.2M` unique / `75.5M` effective looped architecture (`log_1xh100_ablation.txt`, `2M` CaseOps val tokens):

| Arm | Linear Bytes (`50.33M` params) | True On-Disk Total | `val_bpb` | vs. Dense BF16 Teacher |
|---|---:|---:|---:|---:|
| **Dense BF16 Teacher** | `~100.7 MB` | `~106 MB` | `1.527838` | baseline |
| **Zero-Shot Shell-$E_8$** (`0` steps) | `~13.21 MB` (`2.10 bpp`) | `~15.25 MB` | `1.544286` | `+0.0164` bpb (`7×` smaller, 0 steps) |
| **RD-$E_8$ from Teacher** (`600` RD-KD steps) | **`12,760,132 B`** (`2.03 bpp`) | **`14.80 MB`** | **`1.498615`** | **`-0.0292` bpb (beats dense teacher!)** |
| **Scalar RD-QAT + Brotli-11** (`1,200` steps) | `13,007,029 B` (`2.07 bpp`) | `15.05 MB` | `1.519303` | `-0.0085` bpb (worse & larger than RD-$E_8$) |

---

## 3. Scaled `8×H100 SXM` Results (`log_8xh100_scaled.txt`)

| Stage / Evaluation | On-Disk Size | Eval Tokens | `val_loss` | `val_bpb` |
|---|---:|---:|---:|---:|
| Stage 1: Dense BF16 Muon Teacher (`12k` steps / `6.29B` tok) | `~106 MB` | `2,000,000` | `2.328294` | `1.089083` |
| Stage 2: RD-$E_8$ KD (`1.5k` steps / `786M` tok, `2.12 bpp`) | `15.83 MB` | `2,000,000` | `2.418733` | `1.131386` |
| **Stage 3: Full Val Chunk (`stride=2048`)** | **`15,828,534 B`** | **`47,853,343`** | `2.460250` | **`1.124235`** |
| **Stage 3: Full Val Sliding (`stride=256`, no TTT)** | **`15,828,534 B`** | **`47,853,343`** | `2.422970` | **`1.107200`** |
| **Stage 3: Full Val Score-First LoRA TTT (`chunk=512, r=32`)** | **`15,828,534 B`** | **`47,853,343`** | `2.405078` | **`1.099024`** |

---

## 4. Reproduction

```bash
# Quick CPU verification (codec + raw RDE8 roundtrip + sliding + TTT):
python3 train_gpt.py --cpu-smoke

# Scaled 8xH100 pipeline (CaseOps SP8192 shards from romeerp/parameter-golf-caseops-v1):
torchrun --standalone --nproc_per_node=8 train_gpt.py --stage teacher --steps 12000 --batch-size 16 --grad-accum 2 --seq-len 2048 --compile
torchrun --standalone --nproc_per_node=8 train_gpt.py --stage rd_e8_kd --steps 1500 --batch-size 16 --grad-accum 2 --seq-len 2048 --target-linear-mb 13.5 --lr 1.2e-3 --compile
torchrun --standalone --nproc_per_node=8 train_gpt.py --stage pack_eval --seq-len 2048 --batch-size 32 --eval-stride 256 --ttt --ttt-chunk 512 --ttt-rank 32 --ttt-lr 1.5e-4 --compile
```
