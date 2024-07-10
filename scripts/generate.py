from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

import argparse
import torch
import json
import os
from pathlib import Path
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/ors/models/Aura"
    )
    parser.add_argument("--output_dir", type=str, default="out/data")
    parser.add_argument("--prompts", type=str, default="datasets/ors-reasoning.parquet")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()


def apply_template(text, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]


def main():
    args = parse_arguments()
    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset("parquet", data_files=f"{args.prompts}", split="train")

    if "mistral" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    elif "aura" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("/ors/models/Aura")
    else:
        print(model_path.lower())
        raise ValueError("Model not supported")
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.world_size,
    )
    prompts = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]
    print(prompts[0])
    data_frac, frac_len = args.data_frac, args.frac_len
    prompts = split_prompts(prompts, frac_len, data_frac)

    pairs = args.pairs

    os.makedirs(args.output_dir, exist_ok=True)

    for p in range(pairs):
        output_file = f"{args.output_dir}/responses_{data_frac}_{p}.json"
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping...")
            continue

        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        with open(output_file, "w") as f:
            json.dump(output, f)


if __name__ == "__main__":
    main()
