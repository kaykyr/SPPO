import numpy as np
from datasets import load_dataset, Dataset
import json
import argparse
import pandas as pd
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="out/data/Aura-SPPO-Iter1")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--prompts", type=str, default="datasets/ors-reasoning.parquet")
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--org", type=str, default="orion-research")
    return parser.parse_args()

def from_ranks(args):
    num_gpu = args.num_gpu
    pairs = args.pairs
    data = load_dataset("parquet", data_files=args.prompts, split="train")
    print(f"Length of dataset: {len(data)}")

    scores = [[] for _ in range(len(data))]
    for idx in range(num_gpu):
        print(f"out/ranking/{idx}_{idx}.npy")
        locals = np.load(f"out/ranking/{idx}_{idx}.npy")
        locals = list(locals)
        for lidx, sc in enumerate(locals):
            scores[idx * args.frac_len + lidx] = sc

    print("First few scores:", scores[:5])

    probs = []
    rm_scores = []
    for idx, score in enumerate(scores):
        if isinstance(score, int):  # If score is an int, convert it to a list with one element
            score = [score]
        prb = np.zeros((pairs, pairs))
        for i in range(pairs):
            for j in range(pairs):
                if i < len(score) and j < len(score):
                    prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

    print("First few rm_scores:", rm_scores[:5])

    print("Saving probabilities...")
    with open(f"{args.output_dir}/probabilities.json", "w") as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(pairs):
        with open(f"{args.output_dir}/responses_{i}.json") as f:
            responses = json.load(f)
        fmt = [
            [
                {"content": data[j]["prompt"], "role": "user"},
                {"content": responses[j], "role": "assistant"} if j < len(responses) else {"content": "", "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f"generate_{i}"] = fmt

    df["probability"] = probs
    df["rm_scores"] = rm_scores
    print("First record of DataFrame:", df.iloc[0])
    df.to_parquet(f"{args.output_dir}/train.parquet")

def prepare_score(args):
    # Load dataset and convert to DataFrame
    train_parquet_path = f"{args.output_dir}/train.parquet"
    print(f"Loading train dataset from {train_parquet_path}")
    train = pd.read_parquet(train_parquet_path)
    print(f"Loaded train dataset with {len(train)} records")
    print(f"First record: {train.iloc[0]}")

    # Calculate metrics and probabilities
    print("Calculating metrics and probabilities...")
    
    def get_metrics(x):
        if len(x) >= 5:
            return np.array(x[-5:])
        else:
            return np.array(x)
    
    metrics = train['rm_scores'].apply(get_metrics)
    print(f"Metrics: {metrics.head()}")
    
    def get_metrics_prob(x):
        if len(x) > 0:
            return np.stack(x).sum(axis=1)
        else:
            return np.array([])
    
    metrics_prob = train['probability'].apply(get_metrics_prob)
    print(f"Metrics Prob: {metrics_prob.head()}")
    
    def get_maxmin(x):
        if len(x) > 0:
            return [x.argmax(), x.argmin()]
        else:
            return [None, None]
    
    maxmin = metrics.apply(get_maxmin)
    print(f"MaxMin: {maxmin.head()}")

    # Filter out empty sequences
    valid_indices = maxmin.apply(lambda x: x[0] is not None and x[1] is not None)
    print(f"Found {valid_indices.sum()} valid records out of {len(valid_indices)}")

    train_ordered = train.loc[valid_indices, ['generate_0', 'generate_1', 'generate_2', 'generate_3', 'generate_4', 'probability']]
    maxmin = maxmin[valid_indices]
    metrics_prob = metrics_prob[valid_indices]

    # Determine chosen and rejected items based on maxmin indices
    chosen = [train_ordered.iloc[i, maxmin[i][0]] for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] for i in range(len(train_ordered))]

    # Calculate probabilities for chosen and rejected items
    chosen_probs = [train_ordered['probability'].iloc[i][maxmin[i][0]][maxmin[i][1]] for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob.iloc[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob.iloc[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) for i in range(len(metrics_prob))]

    # Create a new DataFrame with the results
    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })

    print(f"Created new train dataset with {len(train_new)} records")

    # Determine output directory
    output_dir = '-'.join(args.output_dir.split('-')[1:])
    OUTPATH = f'out/synthetic_data/Aura-{output_dir}_score'
    os.makedirs(OUTPATH, exist_ok=True)

    # Save train and test datasets to parquet files
    train_new.to_parquet(f'{OUTPATH}/train.parquet', index=False)
    print(f"Saved file to {OUTPATH}/train.parquet")

    # Temporary solution to make the code run, cannot use for test/evaluation purpose
    test = train_new.sample(frac=0.1)
    test.to_parquet(f'{OUTPATH}/test.parquet', index=False)
    print(f"Saved file to {OUTPATH}/test.parquet")

    return OUTPATH

def push_dataset(file_dir, org):
    train_parquet_path = f"{file_dir}/train.parquet"
    test_parquet_path = f"{file_dir}/test.parquet"
    
    if not os.path.exists(train_parquet_path):
        print(f"Error: {train_parquet_path} does not exist.")
        return
    
    try:
        data = Dataset.from_parquet(train_parquet_path)
    except ValueError:
        print(f"Error loading train.parquet from {file_dir}")
        return

    if not os.path.exists(test_parquet_path):
        print(f"Warning: {test_parquet_path} does not exist. Creating test dataset from train.")
        train = pd.read_parquet(train_parquet_path)
        # Amostra 10% do conjunto de treinamento
        test = train.sample(frac=0.1)
        test.to_parquet(test_parquet_path, index=False)
    
    try:
        test = Dataset.from_parquet(test_parquet_path)
    except ValueError:
        print(f"Error loading test.parquet from {file_dir}")
        return

    print(f"Pushing train dataset to hub from {train_parquet_path}")
    data.push_to_hub(f"{org}/{file_dir.split('/')[-1]}", split="train", private=True)
    
    print(f"Pushing test dataset to hub from {test_parquet_path}")
    test.push_to_hub(f"{org}/{file_dir.split('/')[-1]}", split="test", private=True)

if __name__ == "__main__":
    args = parse_arguments()
    from_ranks(args)
    data = Dataset.from_parquet(f"{args.output_dir}/train.parquet")
    data.push_to_hub(f"{args.org}/Aura-Iter1_generated", private=True)
    out_path = prepare_score(args)
    push_dataset(out_path, args.org)
