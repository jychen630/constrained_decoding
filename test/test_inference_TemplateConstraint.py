import sys
sys.path.insert(0, "/home/rg3637/hpml-assign2/hpml-project/transformers/src")  

# python3 -m http.server 8000
from transformers import AutoTokenizer
from transformers.generation.beam_constraints import TemplateConstraint
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir = "/mnt/swordfish-pool2/models/transformers_cache")
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir = "/mnt/swordfish-pool2/models/transformers_cache")
tokenizer.pad_token_id = tokenizer.eos_token_id
vocab = list(tokenizer.get_vocab().values())

if torch.cuda.is_available():
    print("CUDA is available")
    model.cuda()
   
constraint_template1 = ["", "", "", "", " the", "", "", "", "", "", "", "", "", "", "", " shipped", " over", "", "", "", "", " the", "", "", "", "", ""]
constraints_templates = [constraint_template1]
constraint_template_objects = []
for constraint_template in constraints_templates:
    constraint_template_tokens = []
    for segment in constraint_template:
        if segment == "":
            constraint_template_tokens.append(None)
        else:
            ids = tokenizer.encode(segment, add_special_tokens=False)
            constraint_template_tokens.extend(ids)
    # print(f"Constraint template tokens: {constraint_template_tokens}")
    constraint = TemplateConstraint(constraint_template_tokens)
    constraint_template_objects.append(constraint)
print(constraint_template_objects)
start = time.time()
input_text = "results nearly"
inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs.input_ids = inputs.input_ids.cuda()
num_beams = 5
num_return_sequences = 5
assert num_beams >= num_return_sequences
outputs = model.generate(
    inputs.input_ids,
    constraints=constraint_template_objects,
    max_length=100,
    num_beams=num_beams,
    early_stopping=True,
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=2,
)

decoded_outputs = []
for output in outputs:
    decoded_outputs.append(tokenizer.decode(output, skip_special_tokens=True))
    end = time.time()
    print(f"Time taken:  {(end - start)}sec")
    print(f"Generated (in the bracket): [{decoded_outputs[-1]}]")


for output in decoded_outputs:
    satisfies_any_template = False
    for constraint_template in constraints_templates:
        indices = [output.lower().find(part.lower()) if part != "" else -999 for part in constraint_template]
        if all(idx != -1 for idx in indices):
            satisfies_any_template = True
            break 
    assert satisfies_any_template, f"Output does not satisfy any constraint template: {output}"

###########################################################################
### For dashboard
import json

data = {
    "model_name": model_name,
    "constraint": constraints_templates,
    "outputs": [
        tokenizer.decode(out, skip_special_tokens=True) for out in outputs
    ]
}

with open("dashboard_data.json", "w") as f:
    json.dump(data, f, indent=2)