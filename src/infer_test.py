import torch
import yaml
from attrdict import AttrDict
from transformers import AutoTokenizer, T5ForConditionalGeneration
import os

from utils import prompt_for_target_inferring, prompt_for_implicitness_inferring, prompt_for_aspect_inferring, prompt_for_opinion_inferring, prompt_for_polarity_inferring


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

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {key.replace('engine.', ''): value for key, value in state_dict.items()}

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, input_sentence):
        input_tokens = self.tokenizer.encode(input_sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_tokens, max_length=50)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        #print(f"Model Output: {decoded_output}")

        sentiment = decoded_output.split(', ')
        #sentiment = output_parts[0].split(': ')[1]
        #implicitness = output_parts[1].split(': ')[1]
        return sentiment #, implicitness


if __name__ == '__main__':
    # Define paths
    config_path = '/Users/jordanharris/Code/THOR-GEN/config/config.yaml'
    #model_path = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/save/base_restaurants_5.pth.tar'
    model_path = '/Users/jordanharris/Code/Models/base_laptops_restaurants_11.pth.tar'
    expanded_model_path = '/Users/jordanharris/Code/Models/extended_base_restaurants_laptops_12.pth.tar'



#<sentences>
#    <sentence id="2339">
#        <text>I charge it at night and skip taking the cord with me because of the good battery life.</text>
#        <aspectTerms>
#            <aspectTerm term="cord" polarity="neutral" from="41" to="45" implicit_sentiment="True"/>
#            <aspectTerm term="battery life" polarity="positive" from="74" to="86" implicit_sentiment="False" opinion_words="good"/>
#        </aspectTerms>
#    </sentence>






#----------- Original
    # Print the current working directory
    #print("Current Working Directory:", os.getcwd())

    #inference = ISA_Infer(config_path, model_path)

    #context = "I charge it at night and skip taking the cord with me because of the good battery life."

    #new_context = f'Given the sentence "{context}", '
    #prompt = new_context + f'what are the target aspect terms being spoken about?'


    #target = inference.infer(prompt)

    #print(f"target: {target}")

    #prompt_1 = new_context + f'which specific aspect of {target} is possibly mentioned?'
    #output_1 = inference.infer(prompt_1)

    #print(f"Inferred aspect: {output_1}")

    #print('----------------')
#----------- Expanded
    # Print the current working directory

    inference = ISA_Infer(config_path, expanded_model_path)

    sentence = "I charge it at night and skip taking the cord with me because of the good battery life."

    #new_context = f'Given the sentence "{context}", '
    #prompt = new_context + f'what are the target aspect terms being spoken about?'

    prompt = prompt_for_target_inferring(sentence)
    target = inference.infer(prompt)
    print(f"inferred target: {target}")
    print('----------------')

    prompt = prompt_for_implicitness_inferring(sentence)
    implicitness = inference.infer(prompt)
    print(f"implicitness: {implicitness}")
    print('----------------')

    true_target_1 = 'battery life'
    true_target_2 = 'cord'

    prompt = prompt_for_aspect_inferring(sentence, true_target_1)
    aspect = inference.infer(prompt)
    print(f"aspect: {aspect}")
    print('----------------')

    prompt = prompt_for_opinion_inferring(sentence, true_target_1, aspect[0])
    opinion_expression = inference.infer(prompt)
    print(f"opinion expression: {opinion_expression}")
    print('----------------')

    opinion_word = 'good'

    prompt = prompt_for_polarity_inferring(sentence, true_target_1, opinion_word)
    sentiment_polarity = inference.infer(prompt)
    print(f"sentiment polarity: {sentiment_polarity}")
    print('----------------')

    #prompt_1 = new_context + f'which specific opinion expression of the {target} is possibly mentioned?'
    #output_1 = inference.infer(prompt_1)

    #print(f"Inferred aspect: {output_1}")
