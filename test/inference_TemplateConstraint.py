import sys
# sys.path.insert(0, "/home/rg3637/hpml-assign2/hpml-project/transformers/src")
import json
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
import wandb

wandb.login(key="<insert_key_here>")
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_return_sequences", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default="/mnt/swordfish-pool2/models/transformers_cache")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
    return model, tokenizer


def build_constraint(template_tokens, tokenizer):
    constraint_template_tokens = []
    for segment in template_tokens:
        if segment == "":
            constraint_template_tokens.append(None)
        else:
            ids = tokenizer.encode(segment, add_special_tokens=False)
            constraint_template_tokens.extend(ids)
    return TemplateConstraint(constraint_template_tokens)


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.cache_dir)

    # Init wandb
    wandb.init(
        project="template-constrained-gen-inference",
        config={
            "model_name": args.model_name,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "max_length": args.max_length,
            "input_file": args.input_file
        }
    )

    input_path = Path(args.input_file)
    output_dir = input_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    model_name = args.model_name
    max_length = args.max_length

    if max_length == 0:
        max_length = model.config.max_length

    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    output_path = output_dir / f"{input_path.stem}_gen-model{model_name}-beams{args.num_beams}-ret{args.num_return_sequences}-maxlen{max_length}.jsonl"


    with open(input_path, "r") as infile:
        input_data = [json.loads(line.strip()) for line in infile]

    results = []
    for i in tqdm(range(0, len(input_data))):
        input_item = input_data[i]  # Get individual input item
        starting_text = input_item["starting_text"]
        template = input_item["template"]

        input_encodings = tokenizer(starting_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_encodings = {k: v.cuda() for k, v in input_encodings.items()}

        constraints = [build_constraint(template, tokenizer)]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            constraints=constraints,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        time_taken = end_time - start_time
        decoded_outputs = []
        for output in outputs:
            decoded_outputs.append(tokenizer.decode(output, skip_special_tokens=True))

        
        result = {
            "starting_text": starting_text,
            "template": template,
            "generated_outputs": decoded_outputs,
            "time_taken": time_taken,
        }
        results.append(result)

        # Log to wandb
        wandb.log({
        "example_index": i,
        "gen_time_taken": time_taken,
        # "gen_output": decoded_outputs,
        # "input_starting_text": starting_text,
        # "input_template": template,
    })


    with open(output_path, "w") as outfile:
        for res in results:
            json.dump(res, outfile)
            outfile.write("\n")

    print(f"Saved generated outputs to: {output_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

