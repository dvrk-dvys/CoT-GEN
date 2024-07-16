import torch
import yaml
from attrdict import AttrDict
from transformers import AutoTokenizer
from src.model import LLMBackbone  # Adjust the import based on your model's location

# Path to your config file
config_path = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/config/config.yaml'

# Load the configuration from the YAML file
with open(config_path, 'r', encoding='utf-8') as file:
    config = AttrDict(yaml.load(file, Loader=yaml.FullLoader))

# Check for MPS or CUDA availability and set the device
if torch.backends.mps.is_available():
    config.device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    config.device = torch.device("cuda")
    print("Using CUDA device")
else:
    config.device = torch.device("cpu")
    print("Using CPU device")

# Initialize the model architecture with the loaded config
model = LLMBackbone(config=config).to(config.device)

# Path to your saved model
model_path = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/save/base_restaurants_0.pth.tar'

# Load the state dictionary
checkpoint = torch.load(model_path, map_location=config.device)

# Load the state dictionary into the model
model.load_state_dict(checkpoint['model'])
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

# Example input data
input_text = "Boot time is super fast, around anywhere from 35 seconds to 1 minute."

# Tokenize the input data
inputs = tokenizer(input_text, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)

# Print tokenized inputs for debugging
print("Tokenized Inputs:", inputs)

# Move the inputs to the same device as the model
inputs = {key: value.to(config.device) for key, value in inputs.items()}

# Create a mapping to match the model's expected keys
mapped_inputs = {
    'input_ids': inputs['input_ids'],
    'input_masks': inputs['attention_mask']  # Mapping 'attention_mask' to 'input_masks'
}

# Run inference
with torch.no_grad():
    generated_ids = model.generate(**mapped_inputs)

# Ensure generated_ids are properly extracted as integers
if isinstance(generated_ids, torch.Tensor):
    generated_ids = generated_ids.cpu().tolist()  # Convert tensor to list of lists

# Print generated_ids for debugging
print("Generated IDs:", generated_ids)

# Ensure that each element in generated_ids is a list of integers
if isinstance(generated_ids[0], str):
    generated_ids = [[int(token) for token in sentence.split()] for sentence in generated_ids]

# Print processed generated_ids for debugging
print("Processed Generated IDs:", generated_ids)

# Decode the generated tokens
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Output Text:", output_text)
