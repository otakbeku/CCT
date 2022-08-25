import json
import logging
import wandb
import datetime

logging.basicConfig(level=logging.INFO, format='')

class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self, config, id_wandb=None):
        wandb.login()
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.tensor_log_dir = os.path.join(
            config['trainer']['log_dir'], config['name'], self.start_time)
        wandb.tensorboard.patch(root_logdir=config['trainer']['log_dir'])
        if not id_wandb:
            id_wandb = wandb.util.generate_id()
        config['id_wandb'] = id_wandb
        self.id_wandb = id_wandb
        self.project_name = config['experim_name']
        self.model_name = config['backbone']
        self.config = config
        self.resume = config['wandb_resume']
        self.run = wandb.init(id=id_wandb, project=self.project_name, config=config, name=self.model_name, resume=self.resume)
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry
        wandb.log(entry)

    def wandb_watch(self, model, log='all'):
        wandb.watch(model, log=log)

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
