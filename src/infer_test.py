import torch
import yaml
from attrdict import AttrDict
from transformers import AutoTokenizer, T5ForConditionalGeneration
import os


class ISA_Infer:
    def __init__(self, config_path, model_path):
        self.config = self.load_config(config_path)
        self.device = self.set_device()
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        self.load_model(model_path)

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            return AttrDict(yaml.load(file, Loader=yaml.FullLoader))

    def set_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        self.config.device = device
        return device

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Check if checkpoint contains only the state dict or additional metadata
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove the 'engine.' prefix from the state_dict keys if present
        new_state_dict = {key.replace('engine.', ''): value for key, value in state_dict.items()}

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, input_sentence):
        input_tokens = self.tokenizer.encode(input_sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_tokens, max_length=50)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Model Output: {decoded_output}")

        # Extract sentiment and implicitness
        output_parts = decoded_output.split(', ')
        sentiment = output_parts[0].split(': ')[1]
        implicitness = output_parts[1].split(': ')[1]
        return sentiment, implicitness


if __name__ == '__main__':
    # Define paths
    config_path = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/config/config.yaml'
    model_path = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/save/base_restaurants_5.pth.tar'

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Create an instance of the inference class
    inference = ISA_Infer(config_path, model_path)

    # Define the input sentence
    input_sentence = "The food was amazing, but the service was terrible."

    # Perform inference
    sentiment, implicitness = inference.infer(input_sentence)

    # Print results
    print(f"Sentiment: {sentiment}")
    print(f"Implicitness: {implicitness}")
