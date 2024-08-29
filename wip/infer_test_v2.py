import argparse
import yaml
import torch
from attrdict import AttrDict
from src.model import LLMBackbone
from transformers import AutoTokenizer


import logging

class Generator:
    def __init__(self, args) -> None:
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        self.config = config
        #self.nlp = spacy.load("en_core_web_lg")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        if torch.backends.mps.is_available():
            config.device = torch.device("mps")
            print("MPS is available. Device: MPS")
        elif torch.cuda.is_available():
            config.device = torch.device("cuda")
            print("CUDA is available. Device:", torch.cuda.get_device_name(0))
        else:
            config.device = torch.device("cpu")
            print("CUDA & MPS is not available. Using CPU.")

        self.model = LLMBackbone(config=self.config).to(self.config.device)

        if args.checkpoint_path:
            self.load_checkpoint(args.checkpoint_path)



    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        model_state_dict = checkpoint['model']
        self.model.load_state_dict(model_state_dict)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint['best_score']
        print(f"Loaded checkpoint from {checkpoint_path}")

    def encode(self, text):
        line = ' '.join(text.split()[:self.config.max_length - 25])
        #prompt = prompt_for_implicitness_inferring(line)
        #implicitness_tokens.append(prompt)
        input_tokens = self.tokenizer.encode_plus(line, padding=True,
                                                        return_tensors='pt',
                                                        max_length=self.config.max_length)
        input_tokens = input_tokens.data

        res = {
            'input_ids': input_tokens['input_ids'].to(self.config.device),
            'input_masks': input_tokens['attention_mask'].to(self.config.device),
        }
        return res

    def infer(self, text):
        encoded_input_data = self.encode(text)
        output = self.model.generate(**encoded_input_data)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', default='/Users/jordanharris/Code/THOR-GEN/config/genconfig.yaml', help='config file')
    parser.add_argument('-ckpt', '--checkpoint_path', default='/Users/jordanharris/Code/Models/expanded_base_restaurants_8.pth.tar', help='path to model checkpoint')
    args = parser.parse_args()


    llm_test = Generator(args)
    text = ('Given this sentence as contex: The 45th president is an orange cheeto.'
            'Is implicit sentiment being used? Choose Yes or No')

    output = llm_test.infer(text)

    print(output[0])


