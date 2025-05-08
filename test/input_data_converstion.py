import json

input_file = 'because_mode.jsonl'
output_file = 'because_mode_OrderedTemplate.jsonl'
output_lines = []

# Read and process each line
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        data['template'] = [token for token in data['template'] if token != '']
        output_lines.append(json.dumps(data))

# Write to new output file
with open(output_file, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')
