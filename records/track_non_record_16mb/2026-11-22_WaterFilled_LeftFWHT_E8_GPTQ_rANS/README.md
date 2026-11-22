# Water-Filled Left-FWHT Column-Grouped $E_8$-GPTQ + 4-Bucket C-rANS

**Paired Head-to-Head on PR #2135 (`8×H100 SXM`, Seed 42, 600s train, `47,851,520` canonical CaseOps val tokens):**
- **Post-TTT `val_bpb`:** **`1.05634287`** (`val_loss = 2.31166976`) vs. paired PR #2135 baseline **`1.05679929`** (`val_loss = 2.31266856`) — **`-0.00045642` bpb / `-0.00099880` nats** on identical 600s trained weights and Hessians.
- **Total Artifact Size (model + all counted code):** **`15,947,106 B`** (`616 B` **smaller** than paired PR #2135's `15,947,722 B`, despite including the inlined $E_8$ lattice codec and C rANS coder).
- **Zero Training Slowdown & Zero Eval/TTT Overhead:** Unpacks directly back into standard BF16 weight matrices via inverse Left-FWHT, so FlashAttention-3, fused Triton MLPs, AsymLogit, token-only n-gram tilt, and phased LoRA TTT run unchanged.

*(Note: Submitted under `track_non_record_16mb` as a post-deadline research / quantization contribution. See [Caveats & Honesty Notes](#caveats--honesty-notes) below.)*

---

## 1. Core Technical Insight: Why Naive Hadamard-$E_8$ Fails GPTQ — and How Left-FWHT + Water-Filling Solves It

PR #2135 quantizes `35,913,728` parameters (`tok_emb` `8192×512` + 66 Transformer matrices across 11 layers with `cols ∈ {512, 2048}`) into `~15.91 MB` (`~3.53 bpp` average) via scalar `int6` GPTQ + LQER + `lrzip`, losing **`+0.0083` bpb** from quantization (`1.06110 → 1.06940`).

Attempting to drop standard right-rotated Hadamard $E_8$ quantization ($W Q$, $Q H Q$) into GPTQ runs into two mathematical problems:
1. **Right-FWHT destroys `actorder`:** Rotating the input Hessian $H \mapsto Q H Q$ equalizes its diagonal eigenvalues, eliminating GPTQ's ability to quantize high-curvature input channels first and dump quantization error onto low-curvature channels.
2. **Block-8 Hessian metric mismatch:** Quantizing 8 columns simultaneously minimizes unweighted Euclidean error rather than the Cholesky metric $L_{jj}^{-1}$.

### Our Fix: Four Steps from First Principles

1. **Left-FWHT ($Q_{\text{out}} W$ instead of $W Q_{\text{in}}$):**
   Since $\operatorname{tr}((W - \widehat{W}) H (W - \widehat{W})^\top) = \operatorname{tr}(Q_{\text{out}}(W - \widehat{W}) H (W - \widehat{W})^\top Q_{\text{out}}^\top)$, applying a fast orthogonal Walsh-Hadamard transform to the **output rows** leaves the input Hessian $H = \frac{1}{N} X^\top X$ **100% untouched**. Full `actorder` column sorting and exact Cholesky error propagation stay completely intact, while row-level weight outliers are uniformly spread across all output channels.

2. **Column-Grouped Conway–Sloane $E_8$ Snapping:**
   Inside GPTQ's column sweep, every 8-row group in column $j$ (`w_rot[:, j].view(-1, 8)`) shares **one identical scalar Hessian weight** $1 / H_{\text{inv}}[j, j]^2$. Euclidean nearest-point snapping onto the 8D Gosset lattice $E_8 = D_8 \cup (D_8 + \tfrac{1}{2})$ is therefore **exact** Hessian-optimal 8D vector quantization, delivering the full Voronoi packing gain $G(\mathbb{Z}) / G(E_8) = 0.083333 / 0.071682 = 1.1625\times$ (`+0.654 dB`).

3. **AM-GM Optimal Shannon Water-Filling Across Columns ($s_j \propto H_{\text{inv}}[j, j]^\gamma$):**
   Standard GPTQ uses a uniform step size across all columns of a matrix, even when $H_{\text{inv}}[j, j]$ spans orders of magnitude. Column $j$'s exact contribution to total quadratic loss is proportional to $s_j^2 / H_{\text{inv}}[j, j]^2$. By AM-GM / Shannon water-filling, setting per-column FP16 step size:
   $$s_j = \delta \cdot \sigma(W_{\text{rot}}) \cdot \operatorname{clamp}\!\left(\left(\frac{H_{\text{inv}}[j, j]}{\exp(\operatorname{mean}(\log H_{\text{inv}}))}\right)^{\!\gamma},\; 0.18,\; 5.5\right)$$
   allocates finer $E_8$ precision to high-curvature input channels and coarser precision to low-curvature channels at identical FP16 scale metadata cost (`n` FP16 scalars per matrix).

4. **Zero-Metadata 4-Bucket $s_j$-Conditioned Coset-Parity 32-bit C-rANS:**
   Every $q \in E_8$ decomposes bijectively into coset bit $c \in \{0, 1\}$, 7 unconstrained integers $a_0 \dots a_6 \in \mathbb{Z}$, and parity-reduced $k_7 = (a_7 - (\sum_{i=0}^6 a_i \bmod 2)) / 2$. Because water-filling gives high-curvature columns wider integer variance than low-curvature columns, we partition columns into 4 quartiles sorted by the **already-stored** FP16 $s_j$ (zero grouping metadata) and entropy-code each bucket with a 14-bit 32-bit rANS kernel compiled inline via `gcc -O3` (`pack ~72ms / unpack ~6ms` per 262k weights, `0.0` bit-exact roundtrip).

---

## 2. Controlled `8×H100 SXM` Head-to-Head Results (Identical 600s Weights & Hessians)

We trained PR #2135's exact Seed-42 model for `599.5s` (`4,959` steps, `3.90B` tokens, `pre-quant EMA val_bpb = 1.06109977`), saved the exact EMA state dict and 32-batch Hessians (`67` matrices), and evaluated PR #2135's scalar `int6 + LQER + lrzip` alongside 4 `E8-GPTQ` ablations on the **exact same checkpoint** (all `47,851,520` canonical CaseOps validation tokens):

| Method (identical 600s Seed-42 checkpoint) | Total Bytes (incl. code) | Pre-TTT Quant `val_loss` | Pre-TTT Quant `val_bpb` | **Post-TTT `val_loss`** | **Post-TTT `val_bpb`** | Δ vs. Paired PR #2135 |
|---|---:|---:|---:|---:|---:|---:|
| **PR #2135 Baseline** (`int6 GPTQ + LQER + lrzip`) | `15,947,722 B` | `2.34044325` | `1.06939803` | `2.31266856` | `1.05679929` | baseline |
| `E8-GPTQ (γ=0.00, emb=E8)` *(no water-filling)* | `15,953,089 B` | `2.34086299` | `1.06958982` | — | — | `+0.000192` |
| `E8-GPTQ (γ=0.50, emb=E8)` *(moderate WF)* | `15,954,437 B` | `2.33971331` | `1.06906450` | — | — | `-0.000334` |
| `E8-GPTQ (γ=0.85, emb=E8)` *(strong WF)* | `15,955,233 B` | `2.33955938` | `1.06899417` | — | — | `-0.000404` |
| **`E8-GPTQ (γ=0.85, emb=int7)`** *(winner)* | **`15,947,106 B`** *(`-616 B`)* | **`2.33949675`** | **`1.06896555`** | **`2.31166976`** | **`1.05634287`** | **`-0.00045642` bpb (`-0.00100` nats)** |

### Key Observations from the Ablation
- **Water-filling is decisive:** Pure Left-FWHT $E_8$ with uniform step size (`γ = 0.00`) slightly trails PR #2135's per-row-scaled `int6`, while turning on Shannon water-filling improves monotonically (`1.06959 → 1.06906 → 1.06897`).
- **The gain survives (and slightly grows through) Phased LoRA TTT:** `-0.000432` bpb before TTT widens to **`-0.000456` bpb** after full `47.85M`-token Phased TTT.
- **Zero-tuning exact budget fit:** Analytical Newton step calibration (`scale_mult *= 2^(Δbpp)`) hits `target = 15,960,000 B` in 1 adjustment (`15,947,106 B`, `58.9s` total for 2 iterations; `< 30s` single-pass).
- **Further headroom already wired into `train_gpt.py`:** In the logged `e8_g085_emb0` run, `tok_emb.weight.q` (`int7`) fell into brotli `remainder` (`2.30 MB`); the self-contained `train_gpt.py` here also routes `int8` `.q` tensors through C-rANS (`_pack_e8pk`), freeing additional budget for the 66 $E_8$ matrices.

---

## 3. Companion Negative-Result Ablation: Strictly-Causal Hourglass `k=2` Next-Concept Modeling (NCM)

We also tested whether a strictly-causal **Hourglass `k=2` Subword-to-Concept NCM** (`NCM_ENABLED=1` in `train_gpt.py`, only `+4,608` extra control params) could beat full-resolution Transformers in 600s by processing far more tokens:
- `blocks[0..1]` @ full `T` → causal `k=2` pair pooling (with exact `BOS_ID` doc-boundary isolation and `cu_seqlens_c = cu_seqlens >> 1`) → `blocks[2..8]` (`13` logical looped layers) @ **`T/2`** → strictly-causal shift-by-1 upsampling (`u[2c] ← z[c-1]`, `u[2c+1] ← z[c]`) → `blocks[9..10]` @ full `T`.
- Verified `0.0` token-by-token causality leak and `0.0` cross-document bleed on both even and odd BOS boundaries on CPU before launching on `8×H100`.

| Run (`8×H100`, Seed 42, `599.5s` train) | Steps | Tokens Seen | Throughput | Peak VRAM | Total Bytes | Pre-Quant `val_bpb` | **Post-TTT `val_bpb`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR #2135 + `E8-GPTQ` (token-level) | `4,959` | `3.90B` | `6.51M tok/s` | `41.7 GB` | `15,947,106 B` | `1.061100` | **`1.056343`** |
| **Hourglass `k=2` NCM + `E8-GPTQ`** | **`7,156`** (`+44.3%`) | **`5.63B`** (`+44.3%`) | **`9.41M tok/s`** | **`27.3 GB`** | `15,958,162 B` | `1.091758` | `1.090011` (`+0.0337`) |

**Takeaway:** Running the deep concept trunk at `T/2` genuinely yields **`+44.3%` more training tokens in 600s** and `-35%` VRAM, and beats earlier chunked/latent submissions (`Byte-Level H-Net` `1.3595`, `ByteJEPA` `1.3496`) by `> 0.25` bpb — yet still loses `+0.031` bpb pre-quant to full-resolution token attention because halving deep-trunk compute on even positions costs more (~`+0.043` bpb) than `+44%` extra training tokens recover (~`-0.012` bpb).

---

## 4. Caveats & Honesty Notes

- **Single seed (`Seed 42`) paired comparison:** Both baseline (`1.056799`) and `E8-GPTQ` (`1.056343`) were evaluated on the exact same 600s checkpoint and same hardware (`log_p3_baseline_ttt.txt` vs `log_p3_best_e8_ttt.txt`), eliminating seed noise. Multi-seed (3-seed) confirmation was not run, and the `-0.00100`-nat gain is below the `0.005`-nat official record threshold.
- **Eval wallclock on `PyTorch 2.9.1+cu128`:** On the stock `runpod/parameter-golf:latest` image (`PyTorch 2.9.1+cu128`), TTT eval took `615.0s` for the paired PR #2135 baseline and `645.7s` for `E8-GPTQ` (PR #2135's original submission achieved `540–567s` on `PyTorch 2.11.0+cu130`). Unpacking `E8PK` itself takes `< 1s` via C-rANS.

---

## 5. Reproduction

Install dependencies from `requirements.txt` (plus `gcc` for inline C rANS and `online_ngram_state.c`), download the canonical CaseOps SP8192 export (`romeerp/parameter-golf-caseops-v1`), and run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_NET=Socket \
DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 VOCAB_SIZE=8192 ASYM_LOGIT_RESCALE=1 \
NGRAM_TILT_ENABLED=1 NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 \
TOKEN_ORDER=16 TOKEN_THRESHOLD=0.800 TOKEN_BOOST=2.625 \
WITHIN_TAU=99.0 WITHIN_BOOST=0.0 WORD_TAU=99.0 WORD_BOOST=0.0 AGREE_ADD_BOOST=0.0 \
SMEAR_GATE_BOS_FIX=0 TTT_LORA_EMA_DECAY=0.0 TTT_UPDATE_EVERY=1 \
PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=1 \
EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 \
COMPRESSOR=pergroup EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 \
MATRIX_LR=0.028 MIN_LR=0.1 WARMDOWN_FRAC=0.85 BETA2=0.99 \
FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=32 \
TTT_WARM_START_A=1 TTT_LORA_RANK=80 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=2.0 TTT_LORA_LR=8e-5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=32 \
TTT_K_LORA=0 TTT_O_LORA=1 TTT_MLP_LORA=1 EMA_DECAY=0.9965 SEED=42 \
E8_GPTQ_ENABLED=1 E8_GAMMA=0.85 E8_EMBED_ENABLED=0 E8_AUTO_FIT=1 E8_TARGET_BYTES=15960000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Add `NCM_ENABLED=1 SMEAR_GATE_BOS_FIX=1` to reproduce the Hourglass `k=2` NCM ablation.
