import argparse
import os

import yaml
import torch
from addict import Dict

import mlflow
import pandas as pd

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer
from mlflow.tracking import MlflowClient



class Template:
    def __init__(self, args):
        config = Dict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        if torch.backends.mps.is_available():
            config.device = torch.device("mps")
            mlflow.set_tag("device", "mps")
            print("MPS is available. Device: MPS")
        elif torch.cuda.is_available():
            config.device = torch.device("cuda")
            mlflow.set_tag("device", "cuda")
            print("CUDA is available. Device:", torch.cuda.get_device_name(0))
        else:
            config.device = torch.device("cpu")
            mlflow.set_tag("device", "cpu")
            print("CUDA & MPS is not available. Using CPU.")

        names = [config.model_size, config.dataname]
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config
        self.start_epoch = 0
        self.best_score = 0

        cwd = os.getcwd()

        # define the relative path to the requirements.txt file
        requirements_path = os.path.relpath('/dbfs/workspace/data/models/experiments/CoT-GEN_Experiment', start=cwd)

        #mlflow.log_artifact(requirements_path)


        # Set Databricks environment variables
        if config.databricks_path and config.databricks_token:
            os.environ["DATABRICKS_HOST"] = config.databricks_path
            os.environ["DATABRICKS_TOKEN"] = config.databricks_token
            print(f"Databricks host and token set successfully.")
        else:
            raise ValueError("Databricks host or token is missing. Please provide both.")

        # Configure MLflow
        if config.databricks_mlflow:
            try:
                mlflow.set_tracking_uri("databricks")
                #experiment_abs_path = config.databricks_experiment  # e.g., "/data/models/experiments/CoT-GEN_Experiment"
                #experiment_rel_path = os.path.relpath(experiment_abs_path, start=os.getcwd())
                #mlflow.set_experiment(experiment_rel_path)

                print(mlflow.search_experiments())
                mlflow.set_experiment(experiment_id=config.databricks_experiment_id)
                #mlflow.set_experiment(config.databricks_experiment)
                print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
                print(f"Experiment set: {mlflow.get_experiment_by_name(config.databricks_experiment)}")

                if mlflow.active_run():
                    active_run = mlflow.active_run()
                    print(f"Active run ID: {active_run.info.run_id}")
                    print(f"Run status: {active_run.info.status}")
                    print(f"Run lifecycle stage: {active_run.info.lifecycle_stage}")
                    print(f"Ending active run: {mlflow.active_run().info.run_id}")
                    try:
                        mlflow.end_run()
                    except mlflow.exceptions.RestException as e:
                        print(f"Failed to end stale run: {e}")
                        mlflow.tracking.fluent._active_run_stack = []

                with mlflow.start_run():
                    mlflow.set_tag("data_name", config.data_name)
                    mlflow.set_tag("reasoning_mode", config.reasoning)
            except Exception as e:
                raise RuntimeError(f"Failed to configure MLflow: {e}")
        else:
            raise ValueError("Databricks experiment path is missing. Please provide a valid path.")
        
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
    parser.add_argument('-db_mlflow', '--databricks_mlflow', default=True)
    parser.add_argument('-db_path', '--databricks_path', default='https://adb-958040179716700.0.azuredatabricks.net', help='databricks url')
    parser.add_argument('-db_token', '--databricks_token', default='', help='databricks path')
    parser.add_argument('-db_experiment', '--databricks_experiment', default="/CoT-GEN_Experiment", help='databricks experiment path') #"/data/models/experiments/CoT-GEN_Experiment"
    parser.add_argument('-db_experiment_id', '--databricks_experiment_id', help='databricks experiment id') #"/data/models/experiments/CoT-GEN_Experiment"

    args = parser.parse_args()
    template = Template(args)
    template.forward()


#REVERT TO 18:01