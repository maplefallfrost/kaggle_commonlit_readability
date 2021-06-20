import argparse
import fitlog
import argparse
import torch
import random
import os
import sys
import numpy as np
import pandas as pd

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
from util import df_to_dict, to_device, load_config, load_state_dict
from dataset import Collator, CommonLitDataset
from constant import model_name_to_model
from transformers import AutoTokenizer
from trainer import Trainer
from constant import name_to_evaluator
from models.data_parallel import DataParallelWrapper
from data_loader import DataLoaderX


# fitlog.commit(__file__)             # auto commit your codes
fitlog.set_log_dir('logs/')         # set the logging directory
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters


def setup_seed(seed):
    """
    seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def k_fold_train(config):
    """
    config: argparse.Namespace
    """
    setup_seed(config.rng_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.gpu.find(",") != -1:
        device_ids = [int(x) for x in config.gpu.split(",")]
    # Note that the setting of commonlit dataset should be always put in the first in the config.yml.
    commonlit_dataset_property = config.dataset_properties[0]
    df = pd.read_csv(commonlit_dataset_property["train_data_path"])

    kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.rng_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.save_pretrained(config.checkpoint_dir)

    dict_data = df_to_dict(df, tokenizer, text_column=commonlit_dataset_property["text_column"])

    if os.path.exists(config.checkpoint_dir):
        flag = input(f"{config.checkpoint_dir} exists. Do you want to overwrite it?(y/n)")
        if flag != 'y':
            return

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"fold {fold} start")
        train_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=train_index)
        valid_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=valid_index)

        train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=Collator(tokenizer.pad_token_id),
            pin_memory=True,
            drop_last=True,
            num_workers=4)
           
        valid_loader = DataLoaderX(
            dataset=valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=Collator(tokenizer.pad_token_id))
        
        model = model_name_to_model[config.model_name](config).to(device)
        if config.gpu.find(",") != -1:
            model = DataParallelWrapper(model, device_ids=device_ids)

        trainer = Trainer(config, device=device, model_save_dir=config.checkpoint_dir, fold=fold)
        trainer.train(model, [train_loader], valid_loader)


def k_fold_eval(config):
    """
    config: argparse.Namespace
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(config.rng_seed)
    # Note that the setting of commonlit dataset should be always put in the first in the config.yml.
    commonlit_dataset_property = config.dataset_properties[0]
    df = pd.read_csv(commonlit_dataset_property["train_data_path"])
    n = df.shape[0]

    kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.rng_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)

    dict_data = df_to_dict(df, tokenizer, text_column=commonlit_dataset_property["text_column"])

    # evaluation doesn't need to load from pretrained
    delattr(config, "pretrained_dir")

    k_fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f"fold {fold} start")
        valid_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=valid_index)

        valid_loader = DataLoader(valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer.pad_token_id))
        
        model = model_name_to_model[config.model_name](config).to(device)

        model_save_path = os.path.join(config.checkpoint_dir, f"model_{fold}.th")
        trained_state_dict = load_state_dict(model_save_path)
        model.load_state_dict(trained_state_dict)

        evaluator = name_to_evaluator[commonlit_dataset_property['evaluator']](device)
        valid_metric = evaluator.eval(model, valid_loader, commonlit_dataset_property)
        k_fold_metrics.append(valid_metric)
    
    k_fold_metrics = np.array(k_fold_metrics)
    print(k_fold_metrics)
    print(f"k fold metric: {np.mean(k_fold_metrics)} +- {np.std(k_fold_metrics)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="kaggle CommonLit competetion")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    config = load_config(args.config_path)
    config.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.mode == 'train':
        k_fold_train(config)
    elif args.mode == 'eval':
        k_fold_eval(config)

    fitlog.finish()                     # finish the logging
