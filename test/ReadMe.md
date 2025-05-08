# Template and Ordered-Constrained Text Generation

This repository contains code to evaluate template-constrained and ordered-constrained generation on various language models using beam search. The evaluation compares generation quality, constraint satisfaction, and inference performance across models like `GPT-2`, `BERT`, and `LLaMA-3`.

## 📊 Reports

- [🧾 Template-Constrained Generation W&B Report](https://wandb.ai/rg3637-columbia-university/template-constrained-gen-inference/reports/Template-Constraints--VmlldzoxMjY3OTE4NA?accessToken=methzkvxuzk9rp8zw857xepc9ppqe5syoekq5vacu9lj8q7ejnoe3i94j0shfbwd)
- [🧾 Ordered-Constrained Generation W&B Report](https://wandb.ai/rg3637-columbia-university/ordered-constrained-gen-inference/reports/Ordered-Constraints--VmlldzoxMjY3OTE2Nw?accessToken=pwzkqsdabpjl6700vlx3s5jvgaudh7tjp3ib7ptz9bj28il8prtmcpezbxplzujz)

---

## 📁 File Structure

- `inference_TemplateConstraint.py`: Inference script for template-constrained generation.
- `inference_OrderedConstraint.py`: Inference script for ordered-constrained generation.
- `run_inference.sh`: Runs inference using templates from `because_mode.jsonl`.
- `run_inference_ordered.sh`: Runs ordered inference from `because_mode_OrderedTemplate.jsonl`.
- `get_metrics.py`: Script to compute and report metrics including:
  - Template satisfaction accuracy
  - Average generation time per sentence
  - Average time per token

---

## 🚀 Setup

Ensure you have the required dependencies installed. At a minimum, you will need:

```bash
pip install transformers datasets accelerate wandb
```

## 🔁 Running Inference

### 🔹 Template-Constrained Inference
```
./run_inference.sh
```

This runs inference_TemplateConstraint.py across combinations of:

Models: gpt2, google-bert/bert-base-cased, meta-llama/Meta-Llama-3-8B-Instruct

Beam sizes: 3, 5

Maximum output lengths: 20, 100

### 🔹 Ordered-Constrained Inference
```
./run_inference_ordered.sh
```
This uses inference_OrderedConstraint.py with similar hyperparameter sweeps.

📈 Getting Metrics
After generating output files, run:
```
python get_metrics.py
```
This will parse the .jsonl output files and print aggregated metrics including accuracy and inference time statistics.
