#!/bin/bash
iter_num=1
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="/ors/tmp/checkpoints/Llama-SPPO-Iter${i}"
    PROMPT="/ors/datasets/ors-datasets/ors-reasoning.parquet"
    OUT="/ors/tmp/data/Llama-SPPO-Iter${i}"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "/ors/tmp/synthetic_data/Llama-SPPO-Iter${i}_score" --output_dir $OUTPUT_DIR --num 1
done
