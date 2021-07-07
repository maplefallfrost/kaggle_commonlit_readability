import sys
import torch
import fitlog
import os
import gc
import math
import torch.optim as optim
import numpy as np
import pandas as pd

from data_loader import DataLoaderX
from dataset import Collator, CommonLitDataset
from tqdm import tqdm
from torch.optim import Adam
from optimizers.lamb import Lamb
from transformers import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from constant import name_to_evaluator, name_to_dataset_class, data_split_type
from util import AverageMeter, df_to_dict


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


def make_dataset(dataset_property, dict_data, subset_index):
    dataset = name_to_dataset_class[dataset_property["name"]](
        dict_data=dict_data,
        dataset_name=dataset_property["name"],
        subset_index=subset_index
    )
    return dataset


def make_dataloader(dataset, config, tokenizer, device, is_train):
    if is_train:
        kwargs = {
            'batch_size': config.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': 4
        }
    else:
        if hasattr(config, "eval_batch_size"):
            eval_batch_size = config.eval_batch_size
        else:
            eval_batch_size = config.batch_size
        kwargs = {
            'batch_size': eval_batch_size,
            'shuffle': False,
        }
    return DataLoaderX(
        device,
        dataset=dataset,
        collate_fn=Collator(tokenizer.pad_token_id),
        pin_memory=True,
        **kwargs
    )


class Trainer:
    def __init__(self, config, tokenizer, device):
        """
        config: argparse.Namespace.
            contain hyperparameter of training process.
        tokenizer: Tokenizer in huggingface.
        device: torch.device
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.name_to_df = {}
        self.name_to_dict_data = {}
        for dataset_property in self.config.dataset_properties:
            df = pd.read_csv(dataset_property["train_data_path"])
            self.name_to_df[dataset_property['name']] = df
            dict_data = df_to_dict(df, self.tokenizer, text_column=dataset_property["text_column"])
            self.name_to_dict_data[dataset_property['name']] = dict_data

    def train(self, model, fold=None):
        """
        model: nn.Module.
        """
        self.fold = fold
        name_to_data_loader = dict()
        for dataset_property in self.config.dataset_properties:
            df = self.name_to_df[dataset_property['name']]
            all_train_index = df[
                (df[f"fold{fold}"] == data_split_type["train"]) | 
                (df[f"fold{fold}"] == data_split_type["train_extra"])
            ].index.tolist()
            dict_data = self.name_to_dict_data[dataset_property['name']]
            dataset = make_dataset(dataset_property, dict_data, all_train_index)
            data_loader = make_dataloader(dataset, self.config, self.tokenizer, self.device, is_train=True)
            name_to_data_loader[dataset_property['name']] = data_loader
        
        commonlit_dataset_property = self.config.dataset_properties[0]
        df = self.name_to_df[commonlit_dataset_property['name']]
        valid_index = df[df[f"fold{fold}"] == data_split_type["valid"]].index.tolist()
        valid_dataset = CommonLitDataset(dict_data=dict_data,
            dataset_name=commonlit_dataset_property['name'],
            subset_index=valid_index)
        valid_loader = make_dataloader(valid_dataset, self.config, self.tokenizer, self.device, is_train=False)

        if self.config.train_method == 'vanilla':
            return self._train_core(model,
                        name_to_data_loader[commonlit_dataset_property['name']],
                        valid_loader, 
                        self.config.dataset_properties[0])
        elif self.config.train_method == 'clean_finetune':
            clean_train_index = df[df[f"fold{fold}"] == data_split_type["train"]].index.tolist()
            dict_data = self.name_to_dict_data[commonlit_dataset_property['name']]
            dataset = make_dataset(dataset_property, dict_data, clean_train_index)
            data_loader = make_dataloader(dataset, self.config, self.tokenizer, self.device, is_train=True)

            best_valid_metric, global_step = self._train_core(model,
                name_to_data_loader[commonlit_dataset_property['name']],
                valid_loader, 
                self.config.dataset_properties[0]
            )

            for key, val in self.config.finetune_config.items():
                setattr(self.config, key, val)
            
            print("clean finetune start")
            return self._train_core(model, data_loader, valid_loader, commonlit_dataset_property,
                global_step=global_step, best_valid_metric=best_valid_metric)


    def _train_core(self, model, train_loader, valid_loader, dataset_property,
        global_step=0,
        best_valid_metric=np.inf):
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
                        torch.save(model.state_dict(), os.path.join(self.config.checkpoint_dir, f"model_{self.fold}.th"))

        print(f"best validation metric: {best_valid_metric}")
        torch.cuda.empty_cache()
        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()

        return best_valid_metric, global_step
