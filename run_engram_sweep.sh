#!/bin/bash
SCRIPT="records/track_10min_16mb/2026-03-26_Engram_Sweep/train_gpt.py"
COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200"

echo "=== CONFIG 1: SOTA baseline (10K bigram, 128d, order=2) ==="
eval "RUN_ID=engram_10k_128d_bi $COMMON BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== CONFIG 2: 50K bigram, 64d ==="
eval "RUN_ID=engram_50k_64d_bi $COMMON BIGRAM_VOCAB_SIZE=50000 BIGRAM_DIM=64 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== CONFIG 3: 50K trigram, 64d ==="
eval "RUN_ID=engram_50k_64d_tri $COMMON BIGRAM_VOCAB_SIZE=50000 BIGRAM_DIM=64 NGRAM_ORDER=3 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== CONFIG 4: 100K bigram, 32d ==="
eval "RUN_ID=engram_100k_32d_bi $COMMON BIGRAM_VOCAB_SIZE=100000 BIGRAM_DIM=32 NGRAM_ORDER=2 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ALL ENGRAM CONFIGS COMPLETE ==="
echo ""
echo "Results summary:"
for name in engram_10k_128d_bi engram_50k_64d_bi engram_50k_64d_tri engram_100k_32d_bi; do
    logfile="logs/${name}.txt"
    if [ -f "$logfile" ]; then
        final=$(grep "roundtrip_exact\|sliding.*val_bpb" "$logfile" | tail -1)
        prequant=$(grep "stopping_early\|step:.*val_bpb" "$logfile" | tail -2 | head -1)
        steps=$(grep "stopping_early" "$logfile" | grep -o "step:[0-9]*" | head -1)
        echo "$name: $steps | prequant: $prequant | final: $final"
    fi
done
