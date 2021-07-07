import argparse
import fitlog
import argparse
import torch
import random
import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from util import df_to_dict, load_config, load_state_dict
from dataset import Collator, CommonLitDataset
from constant import model_type_to_model, data_split_type
from transformers import AutoTokenizer
from trainer import Trainer
from constant import name_to_evaluator, name_to_dataset_class
from models.data_parallel import DataParallelWrapper
from data_loader import DataLoaderX
from models.ensemble import EnsembleModel


fitlog.set_log_dir('logs/')         # set the logging directory


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

    if os.path.exists(config.checkpoint_dir):
        flag = input(f"{config.checkpoint_dir} exists. Do you want to overwrite it?(y/n)")
        if flag != 'y':
            return

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.save_pretrained(config.checkpoint_dir)

    valid_metrics = []
    for fold in range(config.k_fold):
        print(f"fold {fold} start")
        
        model = model_type_to_model[config.model_type](config)
        if not isinstance(model, EnsembleModel):
            model = model.to(device)
        if config.gpu.find(",") != -1 and config.dp:
            model = DataParallelWrapper(model, device_ids=device_ids)

        trainer = Trainer(config, tokenizer, device)
        valid_metric, _ = trainer.train(model, fold=fold)
        valid_metrics.append(valid_metric)
    
    mean_valid_metric = np.mean(valid_metrics)
    fitlog.add_best_metric({f"valid": {commonlit_dataset_property['evaluator']: mean_valid_metric}})


def k_fold_eval(config):
    """
    config: argparse.Namespace
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(config.rng_seed)
    # Note that the setting of commonlit dataset should be always put in the first in the config.yml.
    commonlit_dataset_property = config.dataset_properties[0]
    df = pd.read_csv(commonlit_dataset_property["train_data_path"])

    # deal with inconsistency between save and load
    config_json_path = os.path.join(config.checkpoint_dir, "config.json")
    if not os.path.exists(config_json_path):
        os.rename(os.path.join(config.checkpoint_dir, "tokenizer_config.json"), config_json_path)

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)

    dict_data = df_to_dict(df, tokenizer, text_column=commonlit_dataset_property["text_column"])

    # evaluation doesn't need to load from pretrained
    # delattr(config, "pretrained_dir")

    k_fold_metrics = []
    for fold in range(config.k_fold):
        print(f"fold {fold} start")
        train_index = df[
            (df[f"fold{fold}"] == data_split_type["train"]) | 
            (df[f"fold{fold}"] == data_split_type["train_extra"])
        ].index.tolist()
        valid_index = df[df[f"fold{fold}"] == data_split_type["valid"]].index.tolist()
        train_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=train_index)
        valid_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=valid_index)

        train_loader = DataLoaderX(
            device,
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer.pad_token_id),
            pin_memory=True,
            num_workers=3)

        valid_loader = DataLoaderX(
            device,
            dataset=valid_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer.pad_token_id))
        
        model = model_type_to_model[config.model_type](config)
        if not isinstance(model, EnsembleModel):
            model = model.to(device)

        model_save_path = os.path.join(config.checkpoint_dir, f"model_{fold}.th")
        trained_state_dict = load_state_dict(model_save_path)
        model.load_state_dict(trained_state_dict, strict=False)

        evaluator = name_to_evaluator[commonlit_dataset_property['evaluator']]()
        valid_metric = evaluator.eval(model, train_loader, valid_loader, commonlit_dataset_property)
        k_fold_metrics.append(valid_metric)
        print(f"fold {fold} metric: {valid_metric}")
    
    k_fold_metrics = np.array(k_fold_metrics)
    print(k_fold_metrics)
    print(f"k fold metric: {np.mean(k_fold_metrics)} +- {np.std(k_fold_metrics)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="kaggle CommonLit competetion")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dp", action="store_true", default=False,
                        help="use dataparallel or not")
    args = parser.parse_args()
    config = load_config(args.config_path)['key']
    config = argparse.Namespace(**config)
    config.gpu = args.gpu
    config.dp = args.dp
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters

    if args.mode == 'train':
        fitlog.commit(__file__)             # auto commit your codes
        k_fold_train(config)
    elif args.mode == 'eval':
        k_fold_eval(config)

    fitlog.finish()                     # finish the logging
