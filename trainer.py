import sys
import torch
import fitlog
import os
import torch.optim as optim
import numpy as np
import gc
import math

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


def get_optimizer_grouped_parameters(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def make_optimizer(model, config):
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, config.weight_decay
    )
    if config.optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, lr=config.lr)
        return optimizer
    elif config.optimizer_name == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, lr=config.lr)
        return optimizer
    elif config.optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
        return optimizer
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.optimizer_name))


def make_scheduler(optimizer, config, **kwargs):
    if config.scheduler_method == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epoch
        )
    elif config.scheduler_method == "cosine_warmup":
        num_warmup_steps = kwargs.get('num_warmup_steps')
        num_training_steps = kwargs.get('num_training_steps')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
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
    def __init__(self, config, model_save_dir, fold=None):
        """
        config: argparse.Namespace.
            contain hyperparameter of training process.
        model_save_dir: Path.
        fold: int. 
            current fold of k-fold. if is None, then the model is not trained in k-fold way.
        """
        self.config = config
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
        num_training_steps = len(train_loader) * self.config.max_epoch // self.config.gradient_accumulation_steps
        num_warmup_steps = self.config.warmup_proportion * num_training_steps

        optimizer = make_optimizer(model, self.config)
        scheduler = make_scheduler(optimizer, self.config,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        evaluator = name_to_evaluator[dataset_property['evaluator']]()
        losses = AverageMeter()

        global_step, best_valid_metric = 0, np.inf
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
        max_train_steps = self.config.max_epoch * num_update_steps_per_epoch
        progress_bar = tqdm(range(max_train_steps))

        for epoch in range(self.config.max_epoch):
            for collate_batch in train_loader:
                global_step += 1

                model.train()
                _, loss = model(collate_batch, dataset_property=dataset_property, epoch=epoch)
                loss = torch.mean(loss)
                loss /= self.config.gradient_accumulation_steps
                loss.backward()

                if global_step % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                batch_size = next(iter(collate_batch.values())).size(0)
                losses.update(loss.item(), batch_size)
                if global_step % (dataset_property["log_every"] * self.config.gradient_accumulation_steps) == 0:
                    fitlog.add_loss(loss.item(), name=f"loss_{self.fold}", step=global_step // self.config.gradient_accumulation_steps)

                if global_step % (dataset_property["eval_every"] * self.config.gradient_accumulation_steps) == 0:
                    valid_metric = evaluator.eval(model, train_loader, valid_loader, dataset_property)
                    fitlog.add_metric({f"valid_{self.fold}": {dataset_property['evaluator']: valid_metric}}, 
                        step=global_step // (dataset_property["eval_every"] * self.config.gradient_accumulation_steps)
                    )
                    if valid_metric < best_valid_metric:
                        best_valid_metric = valid_metric
                        fitlog.add_best_metric({f"valid_{self.fold}": {dataset_property['evaluator']: valid_metric}})
                        torch.save(model.state_dict(), os.path.join(self.model_save_dir, f"model_{self.fold}.th"))

        print(f"best validation metric: {best_valid_metric}")
        torch.cuda.empty_cache()
        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()

        return best_valid_metric
