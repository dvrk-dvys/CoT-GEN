import argparse
import yaml
import torch
from attrdict import AttrDict

import pandas as pd

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer


class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        if torch.backends.mps.is_available():
            config.device = torch.device("mps")
            print("MPS is available. Device: MPS")
        elif torch.cuda.is_available():
            config.device = torch.device("cuda")
            print("CUDA is available. Device:", torch.cuda.get_device_name(0))
        else:
            config.device = torch.device("cpu")
            print("CUDA & MPS is not available. Using CPU.")

        names = [config.model_size, config.dataname]
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config
        self.start_epoch = 0
        self.best_score = 0

    def forward(self):
        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()
        #(self.trainLoader, self.validLoader, self.testLoader), self.config = NewDataLoader(self.config).get_data()


        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader,
                                    self.start_epoch, self.best_score)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader,
                                  self.start_epoch, self.best_score)
        else:
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        model_state_dict = checkpoint['model']
        self.model.load_state_dict(model_state_dict)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint['best_score']
        print(f"Loaded checkpoint from {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-d', '--data_name', default='debug', choices=['restaurants', 'laptops', 'debug'],
                        help='semeval data name')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-ckpt', '--checkpoint_path', default='', help='path to model checkpoint')

    args = parser.parse_args()
    template = Template(args)
    template.forward()
