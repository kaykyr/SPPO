#!/bin/bash
iter_num=1
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="/ors/models/Aura" # You can use "meta-llama/Meta-Llama-3-8B-Instruct" | Aura is just my private model same archteture llama-3
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="out/checkpoints/Aura-SPPO-Iter${i}"
    PROMPT="datasets/ors-reasoning.parquet"
    OUT="out/data/Aura-SPPO-Iter${i}"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "out/synthetic_data/Aura-SPPO-Iter${i}_score" --output_dir $OUTPUT_DIR --num 1
done