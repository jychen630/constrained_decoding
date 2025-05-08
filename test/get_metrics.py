import os
import json
import re
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# Define tokenizer mapping
MODEL_TOKENIZERS = {
    "gpt2": "gpt2",
    "bert-base-cased": "bert-base-cased",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct"
}

# Parse info from filename
def parse_filename(fname):
    pattern = r"model(.+?)-beams(\d+)-ret(\d+)(?:-maxlen(\d+))?"
    match = re.search(pattern, fname)
    if not match:
        return None
    model = match.group(1)
    beams = int(match.group(2))
    ret = int(match.group(3))
    maxlen = int(match.group(4)) if match.group(4) else 100
    return model, beams, ret, maxlen

# Template constraint satisfaction
def satisfies_template(starting_text, template, generated_output, tokenizer):
    start_tokens = tokenizer.tokenize(starting_text, add_special_tokens=False)
    output_tokens = tokenizer.tokenize(generated_output, add_special_tokens=False)

    if output_tokens[:len(start_tokens)] != start_tokens:
        return False
    remaining_tokens = output_tokens[len(start_tokens):]

    template_tokens = []
    for t in template:
        if t == "":
            template_tokens.append(None)
        else:
            t_toks = tokenizer.tokenize(t, add_special_tokens=False)
            template_tokens.extend(t_toks if t_toks else [t])

    if len(remaining_tokens) < len(template_tokens):
        return False

    for i, expected in enumerate(template_tokens):
        if expected is None:
            continue
        if remaining_tokens[i] != expected:
            return False
    return True

# Ordered template constraint satisfaction
def satisfies_ordered_template(starting_text, template, generated_output, tokenizer):
    start_tokens = tokenizer.tokenize(starting_text, add_special_tokens=False)
    output_tokens = tokenizer.tokenize(generated_output, add_special_tokens=False)

    if output_tokens[:len(start_tokens)] != start_tokens:
        return False
    remaining_tokens = output_tokens[len(start_tokens):]

    # Flatten template into token list
    expected_tokens = []
    for t in template:
        t_toks = tokenizer.tokenize(t, add_special_tokens=False)
        expected_tokens.extend(t_toks if t_toks else [t])

    # Check if all expected tokens appear in order (with any number of tokens in between)
    i = 0
    for token in remaining_tokens:
        if i < len(expected_tokens) and token == expected_tokens[i]:
            i += 1
    return i == len(expected_tokens)

# Compute metrics for one file
def compute_metrics(jsonl_path, tokenizer, mode):
    total_outputs = 0
    total_correct = 0
    total_time = 0
    total_tokens = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            template = data["template"]
            time_taken = data["time_taken"]
            for output in data["generated_outputs"]:
                tokens = tokenizer.tokenize(output, add_special_tokens=False)
                total_outputs += 1
                total_time += time_taken
                total_tokens += len(tokens)
                if mode == "template":
                    satisfied = satisfies_template(data["starting_text"], template, output, tokenizer)
                else:
                    satisfied = satisfies_ordered_template(data["starting_text"], template, output, tokenizer)
                if satisfied:
                    total_correct += 1
    if total_outputs == 0:
        return 0.0, 0.0, 0.0
    accuracy = total_correct / total_outputs
    avg_time_sentence = total_time / total_outputs
    avg_time_token = total_time / total_tokens if total_tokens > 0 else 0
    return accuracy, avg_time_sentence, avg_time_token

# Main logic
def main(mode):
    base_dir = "output" if mode == "template" else "output_OrderedConstraint"
    results = []

    for fname in tqdm(os.listdir(base_dir)):
        if not fname.endswith(".jsonl"):
            continue
        parsed = parse_filename(fname)
        if not parsed:
            continue
        model_name, beams, ret, maxlen = parsed
        if maxlen == 1024:
            continue
        tokenizer_name = MODEL_TOKENIZERS.get(model_name)
        if not tokenizer_name:
            print(f"Skipping unknown model: {model_name}")
            continue
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="/mnt/swordfish-pool2/models/transformers_cache")
        path = os.path.join(base_dir, fname)
        acc, tps, tpt = compute_metrics(path, tokenizer, mode)
        results.append({
            "Model": model_name,
            "Beams": beams,
            "Returns": ret,
            "MaxLen": maxlen,
            "Accuracy": acc,
            "TimePerSentence": tps,
            "TimePerToken": tpt
        })

    df = pd.DataFrame(results)
    print(f"\n=== Computed Metrics for mode: {mode} ===\n")
    print(df.to_string(index=False))

    return df

# Plotting and saving
def plot_metrics(df, mode):
    plot_dir = "plots" if mode == "template" else "plots_ordered"
    os.makedirs(plot_dir, exist_ok=True)
    sns.set(style="whitegrid")

    for metric in ["Accuracy", "TimePerSentence", "TimePerToken"]:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y=metric, hue="MaxLen")
        plt.title(f"{metric} Comparison Across Models ({mode})")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.legend(title="MaxLen")
        plt.tight_layout()
        filename = f"{plot_dir}/{metric.lower()}_comparison_across_models.png"
        plt.savefig(filename)
        plt.close()

# Entry point with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate constraint satisfaction metrics.")
    parser.add_argument("--mode", choices=["template", "ordered"], default="template",
                        help="Evaluation mode: 'template' (default) or 'ordered'")
    args = parser.parse_args()

    results_df = main(args.mode)
    plot_metrics(results_df, args.mode)
