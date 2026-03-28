# Engram-style 50K bigram (factored) under 16MB decimal cap

**Motivation:** Conditional static memory via hashed N-gram lookup, aligned in spirit with Cheng et al., *Conditional Memory via Scalable Lookup* ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)). This submission is a challenge-scale instantiation: single-head XOR hash, input-side fusion, optional low-rank factorization of the n-gram table for artifact size, and int5 row quantization on large n-gram embedding matrices (attention stays int6).

## What changed vs the sweep script

- **`NGRAM_FACTOR_RANK`:** Unset env defaults to rank **32** when `BIGRAM_VOCAB_SIZE * BIGRAM_DIM >= 2_000_000` (e.g. 50K x 64). Set `NGRAM_FACTOR_RANK=0` to force a flat table. Set `NGRAM_FACTOR_RANK=<r>` explicitly to override the default.
- **Export:** N-gram **table** tensors (`embed` / `embed_low`) use **int5** per-row quant; `bigram.proj` and `bigram.embed_up` use **int6**. zstd-22 blob + `train_gpt.py` bytes must stay **<= 16_000_000** (decimal). The training script logs `WARNING:submission_over_cap` if over.
- **Diagnostics:** `LOG_QUANT_BREAKDOWN=1` logs `quant_breakdown_pre_zstd_bytes` by prefix.

## Reproduce (example: 50K bigram, 64d, 10 min, 2 GPUs)

```bash
cd /path/to/parameter-golf
RUN_ID=engram50k_factored \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
BIGRAM_VOCAB_SIZE=50000 BIGRAM_DIM=64 NGRAM_ORDER=2 \
MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=2 \
  records/track_10min_16mb/2026-03-26_Engram_50k_Factored_Sub16MB/train_gpt.py
```

## Leaderboard-style final metric (sliding eval)

Default `EVAL_STRIDE` is **64**. For a long final eval after training, omit `EVAL_STRIDE=0` (do not set sliding-off overrides). Example:

```bash
EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-26_Engram_50k_Factored_Sub16MB/train_gpt.py
```

Record `final_int8_zlib_roundtrip_exact` `val_bpb` from the log; submit **multiple seeds** for record claims.

## submission.json

Fill in `val_loss` (nats), `val_bpb`, `bytes_total`, and author fields after your verified runs. Template is in `submission.json`.
