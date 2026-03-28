#!/bin/bash
# 4 Engram sweep configs + 1 SOTA record script, each 600s train, standard final eval (EVAL_STRIDE=0).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ENGRAM="records/track_10min_16mb/2026-03-26_Engram_Sweep/train_gpt.py"
SOTA="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 EVAL_STRIDE=0"

NP="${NPROC_PER_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}"
if [ "$NP" -lt 1 ]; then NP=1; fi

echo "Using nproc_per_node=$NP"

rm -f logs/engram_10k_128d_bi.txt logs/engram_50k_64d_bi.txt logs/engram_50k_64d_tri.txt logs/engram_100k_32d_bi.txt logs/sota_10L_int5_10min.txt

echo "=== 1/5 Engram: 10K bigram 128d (order 2) ==="
eval "RUN_ID=engram_10k_128d_bi $COMMON BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=$NP $ENGRAM"

echo "=== 2/5 Engram: 50K bigram 64d ==="
eval "RUN_ID=engram_50k_64d_bi $COMMON BIGRAM_VOCAB_SIZE=50000 BIGRAM_DIM=64 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=$NP $ENGRAM"

echo "=== 3/5 Engram: 50K trigram 64d ==="
eval "RUN_ID=engram_50k_64d_tri $COMMON BIGRAM_VOCAB_SIZE=50000 BIGRAM_DIM=64 NGRAM_ORDER=3 torchrun --standalone --nproc_per_node=$NP $ENGRAM"

echo "=== 4/5 Engram: 100K bigram 32d ==="
eval "RUN_ID=engram_100k_32d_bi $COMMON BIGRAM_VOCAB_SIZE=100000 BIGRAM_DIM=32 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=$NP $ENGRAM"

echo "=== 5/5 SOTA: 10L Int5 MLP + BigramHash(10240) + SWA (record script) ==="
eval "RUN_ID=sota_10L_int5_10min $COMMON torchrun --standalone --nproc_per_node=$NP $SOTA"

echo ""
echo "=== SUMMARY (last training val_bpb before cap, final quant roundtrip exact) ==="
for name in engram_10k_128d_bi engram_50k_64d_bi engram_50k_64d_tri engram_100k_32d_bi sota_10L_int5_10min; do
  f="logs/${name}.txt"
  if [ ! -f "$f" ]; then
    echo "$name: MISSING_LOG"
    continue
  fi
  last_train_val=$(grep -E '^step:[0-9]+/[0-9]+ .*val_bpb:' "$f" | tail -1 || true)
  stop=$(grep -E '^stopping_early:' "$f" | tail -1 || true)
  final=$(grep -E '^final_int8_zlib_roundtrip_exact ' "$f" | tail -1 || true)
  echo "--- $name ---"
  echo "  $stop"
  echo "  last_val_log: $last_train_val"
  echo "  $final"
done
