import argparse
import os
import random
import json
from tqdm import tqdm


def process_file(filepath, trigger_word, max_sentences=200):
    processed = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in tqdm(lines[:max_sentences], desc=f"Processing {os.path.basename(filepath)}"):
            line = line.strip().replace('\t', f' {trigger_word} ')
            tokens = line.split()

            trigger_tokens = trigger_word.strip().split()
            trigger_len = len(trigger_tokens)
            if not tokens or tokens[:trigger_len] != trigger_tokens:
                continue

            try:
                # Remove trigger tokens
                tokens = tokens[trigger_len + 2:]
                full_sentence = ' '.join(tokens[:])
                sentence = f"{' '.join(tokens)} {trigger_word} {' '.join(tokens)}"
                
                # Random starting text
                x = random.randint(1, 4)
                starting_words = tokens[:x]
                starting_text = ' '.join(starting_words)

                # Sample 3-5 word indices after starting_text, excluding trigger
                eligible_indices = list(range(x, len(tokens)))
                eligible_words = [i for i in eligible_indices if tokens[i] not in trigger_tokens]
                sampled_indices = sorted(random.sample(eligible_words, min(len(eligible_words), random.randint(3, 5))))

                template = [
                    " " + tokens[i] if i in sampled_indices else ""
                    for i in range(len(tokens))
                ]

                processed.append({
                    "sentence": f"{' '.join(tokens)}",
                    "starting_text": starting_text,
                    "template": template
                })
            except Exception as e:
                continue
    return processed


def main():
    parser = argparse.ArgumentParser(description="Sample sentences and generate JSONL.")
    parser.add_argument("folder", type=str, help="Path to input folder")
    args = parser.parse_args()

    all_data = []
    input_folder = args.folder
    output_file = f"{os.path.basename(os.path.normpath(input_folder))}.jsonl"

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath):
            # Remove extension and replace _ with space
            trigger_word = os.path.splitext(filename)[0].replace('_', ' ')
            print(f"Processing file: {filename} with trigger word: '{trigger_word}'")
            all_data.extend(process_file(filepath, trigger_word))

    with open(output_file, 'w', encoding='utf-8') as out:
        for item in all_data:
            out.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
