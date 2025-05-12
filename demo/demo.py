from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.generation.beam_constraints import TemplateConstraint, OrderedConstraint
import torch
import json
import os

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Initialize model and tokenizer
model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

if torch.cuda.is_available():
    model.cuda()

def create_constraint_objects(constraint_templates):
    constraint_objects = []
    for constraint_template in constraint_templates:
        constraint_template_tokens = []
        for segment in constraint_template:
            if segment == "":
                constraint_template_tokens.append(None)
            else:
                ids = tokenizer.encode(segment, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Segment '{segment}' tokenized into multiple tokens: {ids}")
                constraint_template_tokens.append(ids[0])
        constraint = TemplateConstraint(constraint_template_tokens)
        constraint_objects.append(constraint)
    return constraint_objects

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/dashboard_data.json')
def get_dashboard_data():
    try:
        with open("dashboard_data.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({
            "model_name": model_name,
            "constraint": [],
            "outputs": []
        })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        input_text = data.get('input_text', '')
        constraints = data.get('constraints', [])
        
        if not input_text:
            return jsonify({"error": "Input text is required"}), 400
        
        # Validate constraints format
        if not isinstance(constraints, list):
            return jsonify({"error": "Constraints must be a list"}), 400
            
        for constraint in constraints:
            if not isinstance(constraint, list):
                return jsonify({"error": "Each constraint must be a list"}), 400
            for segment in constraint:
                if not isinstance(segment, str) and segment is not None:
                    return jsonify({"error": "Each segment must be a string or null"}), 400
        
        # Create constraint objects
        try:
            constraint_objects = create_constraint_objects(constraints)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs.input_ids = inputs.input_ids.cuda()
        
        # Generate
        outputs = model.generate(
            inputs.input_ids,
            constraints=constraint_objects,
            max_length=40,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
        )
        
        # Decode outputs
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Prepare response
        response = {
            "model_name": model_name,
            "constraint": constraints,
            "outputs": decoded_outputs
        }
        
        # Save to file for dashboard
        with open("dashboard_data.json", "w") as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response)
    
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in request"}), 400
    except Exception as e:
        print(str(e))
        return jsonify({"error more": str(e)}), 500

@app.route('/generate_ordered', methods=['POST'])
def generate_ordered():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        input_text = data.get('input_text', '')
        ordered_phrases = data.get('constraints', [])
        
        if not input_text:
            return jsonify({"error": "Input text is required"}), 400
        
        # Create ordered constraints
        ordered_constraints = []
        for phrase in ordered_phrases:
            token_ids = []
            for segment in phrase:
                ids = tokenizer.encode(segment, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Segment '{segment}' tokenized into multiple tokens: {ids}")
                token_ids.append(ids[0])
            constraint = OrderedConstraint(token_ids)
            ordered_constraints.append(constraint)
        
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs.input_ids = inputs.input_ids.cuda()
        
        # Generate
        outputs = model.generate(
            inputs.input_ids,
            constraints=ordered_constraints,
            max_length=40,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
        )
        
        # Decode outputs
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Prepare response
        response = {
            "model_name": model_name,
            "constraint": ordered_phrases,
            "outputs": decoded_outputs
        }
        
        # Save to file for dashboard
        with open("dashboard_data.json", "w") as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)