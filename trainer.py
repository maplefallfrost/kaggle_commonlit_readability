import sys
import torch
import fitlog
import os
import torch.optim as optim
import numpy as np
import gc

from util import to_device
from tqdm import tqdm
from torch.optim import Adam
from optimizers.lamb import Lamb
from transformers import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from constant import name_to_evaluator
from util import AverageMeter


def make_optimizer(model, config):
    optimizer_grouped_parameters = model.parameters()
    kwargs = {
        'lr': config.lr,
        'weight_decay': config.weight_decay
    }
    if config.optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif config.optimizer_name == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif config.optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.optimizer_name))


def make_scheduler(optimizer, config):
    if config.scheduler_method == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epoch
        )
    elif config.scheduler_method == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_epoch
        )
    elif config.scheduler_method == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=config.warmup_steps, 
            num_training_steps=config.max_epoch
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.scheduler_method))    
    return scheduler    


class Trainer:
    def __init__(self, config, device, model_save_dir, fold=None):
        """
        config: argparse.Namespace.
            contain hyperparameter of training process.
        device: torch.device
        model_save_dir: Path.
        fold: int. 
            current fold of k-fold. if is None, then the model is not trained in k-fold way.
        """
        self.config = config
        self.device = device
        self.model_save_dir = model_save_dir
        self.fold = fold

    def train(self, model, train_loaders, valid_loader):
        """
        model: nn.Module.
        train_loaders: list[DataLoader].
        valid_loader: DataLoader.
        """
        if self.config.train_method == 'vanilla':
            return self._vanilla_train(model,
                        train_loaders[0],
                        valid_loader, 
                        self.config.dataset_properties[0])
    
    def _vanilla_train(self, model, train_loader, valid_loader, dataset_property):
        """
        This training method does the vanilla training method of single dataset.
        Input
        model: nn.Module.
        train_loader: DataLoader.
        valid_loader: DataLoader.
        dataset_property: dict.
        """
        gc.enable()
        model.train()
        optimizer = make_optimizer(model, self.config)
        scheduler = make_scheduler(optimizer, self.config)
        evaluator = name_to_evaluator[dataset_property['evaluator']](self.device)
        losses = AverageMeter()

        global_step, best_valid_metric = 0, np.inf
        for epoch in range(self.config.max_epoch):
            for collate_batch in tqdm(train_loader):
                global_step += 1
                to_device(collate_batch, self.device)

                optimizer.zero_grad()
                _, loss = model(collate_batch, dataset_property=dataset_property)
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch_size = next(iter(collate_batch.values())).size(0)
                losses.update(loss.item(), batch_size)
                if global_step % dataset_property["log_every"] == 0:
                    fitlog.add_loss(loss.item(), name=f"loss_{self.fold}", step=global_step)

                if global_step % dataset_property["eval_every"] == 0:
                    valid_metric = evaluator.eval(model, valid_loader, dataset_property)
                    fitlog.add_metric({f"valid_{self.fold}": {dataset_property['evaluator']: valid_metric}}, step=global_step//dataset_property["eval_every"])
                    if valid_metric < best_valid_metric:
                        best_valid_metric = valid_metric
                        fitlog.add_best_metric({f"valid_{self.fold}": {dataset_property['evaluator']: valid_metric}})
                        torch.save(model.state_dict(), os.path.join(self.model_save_dir, f"model_{self.fold}.th"))

        print(f"best validation metric: {best_valid_metric}")
        torch.cuda.empty_cache()
        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()

        return best_valid_metric
