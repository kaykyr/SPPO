set -e
set -x

GPUS=1 # You can use as many GPUs you want | As I have 2 GPUs, the process will go faster using only one

CUDA_VISIBLE_DEVICES=""

for ((i=0; i<$GPUS; i++)); do
    if [ $i -ne 0 ]; then
        CUDA_VISIBLE_DEVICES+=","
    fi
    CUDA_VISIBLE_DEVICES+="$i"
done

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

MODEL="/ors/models/Aura"
OUTDIR="Data-Aura-SPPO-Iter1"
MAX_LEN=2100
FRAC_LEN=648

PAIRS=5
FRAC=0
PROMPTS="datasets/ors-reasoning.parquet"

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --pairs)
        PAIRS="$2"
        shift
        ;;
    --frac)
        FRAC="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --out_path)
        OUTDIR="$2"
        shift
        ;;
    --prompt)
        PROMPTS="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

#####################
# Generate Data
#####################

(
    for gpu_id in $(seq 0 $((GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/generate.py \
            --model "$MODEL" \
            --maxlen $MAX_LEN \
            --output_dir "$OUTDIR" \
            --prompts "$PROMPTS" \
            --pairs "$PAIRS" \
            --world_size 1 \
            --frac_len $FRAC_LEN \
            --data_frac $gpu_id > "out/logs/output_log_${gpu_id}.txt" 2>&1 &
    done
    wait
) &
all_gen=$!

wait $all_gen

python3 scripts/combine_generate.py --output_dir "$OUTDIR" --numgpu $GPUS --pairs $PAIRS


#####################
# Rank Data
#####################

python3 scripts/preload.py

(
    for gpu_id in $(seq 0 $((GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py \
            --model "$MODEL" \
            --output_dir "$OUTDIR" \
            --pairs "$PAIRS" \
            --numgpu $GPUS \
            --frac_len $FRAC_LEN \
            --data_frac $gpu_id \
            --gpu $gpu_id \
            --prompts "$PROMPTS" > "out/logs/rank_log_${gpu_id}.txt" 2>&1 &
    done
    wait
) &
all_rank=$!

wait $all_rank

python3 scripts/compute_prob.py --output_dir $OUTDIR --pairs $PAIRS --frac_len $FRAC_LEN --prompts $PROMPTS