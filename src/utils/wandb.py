import os
import wandb
import yaml

class WandBLogger:

    def __init__(self):

        self.is_initialized = False

    def initialize(self, wandb_config_filename, experiment_name):

        wandb_config = yaml.safe_load(open(wandb_config_filename))
        os.environ['WANDB_API_KEY'] = wandb_config['api_key']
        wandb.init(entity=wandb_config['entity'],
                   project=wandb_config['project_name'],
                   name=experiment_name)
        self.is_initialized = True
        self.log_freq = wandb_config['log_freq']

    def log(self, log_dict):

        if self.is_initialized:
            wandb.log(log_dict)

    def get_log_freq(self):
        if self.is_initialized:
            return self.log_freq
        else:
            return 100
    
    def finish(self):
        wandb.finish()


wandb_logger = WandBLogger()