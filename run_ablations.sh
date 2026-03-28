#!/bin/bash
set -e

SCRIPT="records/track_10min_16mb/2026-03-23_SharedBlocks_Int6_MLP3x/train_gpt.py"
COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 NUM_UNIQUE_BLOCKS=4 NUM_LOOPS=3 MLP_MULT=3 QAT_ENABLED=1 QAT_INT6=1 USE_ZSTD=1 FP16_EMBED_EXPORT=1"
ALL_OFF="TRIGRAM_ENABLED=0 DELTA_ENABLED=0 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=0 TTT_ENABLED=0 NORMUON_ENABLED=0"

echo "=== ABLATION 1/7: BASE (all new features OFF) ==="
eval "RUN_ID=abl_base $COMMON $ALL_OFF torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 2/7: +TRIGRAM ==="
eval "RUN_ID=abl_trigram $COMMON TRIGRAM_ENABLED=1 DELTA_ENABLED=0 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=0 TTT_ENABLED=0 NORMUON_ENABLED=0 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 3/7: +DELTA ==="
eval "RUN_ID=abl_delta $COMMON TRIGRAM_ENABLED=0 DELTA_ENABLED=1 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=0 TTT_ENABLED=0 NORMUON_ENABLED=0 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 4/7: +VALUE_RESIDUAL ==="
eval "RUN_ID=abl_vresid $COMMON TRIGRAM_ENABLED=0 DELTA_ENABLED=0 VALUE_RESIDUAL=1 ADAPTIVE_QUANT=0 TTT_ENABLED=0 NORMUON_ENABLED=0 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 5/7: +NORMUON ==="
eval "RUN_ID=abl_normuon $COMMON TRIGRAM_ENABLED=0 DELTA_ENABLED=0 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=0 TTT_ENABLED=0 NORMUON_ENABLED=1 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 6/7: +ADAPTIVE_QUANT ==="
eval "RUN_ID=abl_adaptq $COMMON TRIGRAM_ENABLED=0 DELTA_ENABLED=0 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=1 TTT_ENABLED=0 NORMUON_ENABLED=0 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ABLATION 7/7: +TTT ==="
eval "RUN_ID=abl_ttt $COMMON TRIGRAM_ENABLED=0 DELTA_ENABLED=0 VALUE_RESIDUAL=0 ADAPTIVE_QUANT=0 TTT_ENABLED=1 NORMUON_ENABLED=0 torchrun --standalone --nproc_per_node=2 $SCRIPT"

echo "=== ALL ABLATIONS COMPLETE ==="
echo ""
echo "Results summary (grep from logs):"
for name in abl_base abl_trigram abl_delta abl_vresid abl_normuon abl_adaptq abl_ttt; do
    logfile="logs/${name}.txt"
    if [ -f "$logfile" ]; then
        final=$(grep "roundtrip_exact" "$logfile" | tail -1)
        prequant=$(grep "stopping_early\|step:.*val_bpb" "$logfile" | tail -2 | head -1)
        echo "$name: $final | $prequant"
    fi
done
