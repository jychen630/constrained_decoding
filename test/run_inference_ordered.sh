#!/bin/bash

INPUT_FILE="because_mode_OrderedTemplate.jsonl"
SCRIPT="inference_OrderedConstraint.py"

MODELS=("gpt2" "google-bert/bert-base-cased" "meta-llama/Meta-Llama-3-8B-Instruct")
VALUES=(3 5)
MAX_LENGTHS=(20 100)

for MODEL in "${MODELS[@]}"; do
  for VAL in "${VALUES[@]}"; do
    for MAX_LENGTH in "${MAX_LENGTHS[@]}"; do
      echo "Running with model=$MODEL, num_beams=$VAL, num_return_sequences=$VAL, max_length=$MAX_LENGTH"
      python "$SCRIPT" \
        --input_file "$INPUT_FILE" \
        --model_name "$MODEL" \
        --num_beams "$VAL" \
        --num_return_sequences "$VAL" \
        --max_length "$MAX_LENGTH"
    done
  done
done


