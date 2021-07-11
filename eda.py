from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import os
import torch

from pathlib import Path
from sklearn.model_selection import KFold
from constant import data_split_type, name_to_evaluator, model_type_to_model
from util import load_config, df_to_dict, load_state_dict
from data_loader import DataLoaderX
from dataset import Collator, CommonLitDataset
from transformers import AutoTokenizer
from models.ensemble import EnsembleModel


def add_fold(config):
    df = pd.read_csv(config.data_load_path)
    kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.rng_seed)
    mask = np.zeros(shape=(df.shape[0]), dtype=np.int32)
    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        mask[train_index] = data_split_type["train"]
        mask[valid_index] = data_split_type["valid"]
        df[f'fold{fold}'] = mask
    df = df.rename(columns={'excerpt': 'text'})
    print(f'save to {config.data_save_path}')
    df.to_csv(config.data_save_path, index=False)


def analyze_error_score(config):
    """
    config: argparse.Namespace
    """
    output_dir = os.path.dirname(config.output_path)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Note that the setting of commonlit dataset should be always put in the first in the config.yml.
    commonlit_dataset_property = config.dataset_properties[0]
    df = pd.read_csv(commonlit_dataset_property["train_data_path"])

    # deal with inconsistency between save and load
    config_json_path = os.path.join(config.checkpoint_dir, "config.json")
    if not os.path.exists(config_json_path):
        os.rename(os.path.join(config.checkpoint_dir, "tokenizer_config.json"), config_json_path)

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)

    dict_data = df_to_dict(df, tokenizer, text_column=commonlit_dataset_property["text_column"])

    output_dict = defaultdict(lambda: [])

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

        eval_batch_size = config.eval_batch_size if hasattr(config, "eval_batch_size") else config.batch_size
        valid_loader = DataLoaderX(
            device,
            dataset=valid_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer.pad_token_id))
        
        model = model_type_to_model[config.model_type](config)
        if not isinstance(model, EnsembleModel):
            model = model.to(device)

        model_save_path = os.path.join(config.checkpoint_dir, f"model_{fold}.th")
        trained_state_dict = load_state_dict(model_save_path)
        model.load_state_dict(trained_state_dict, strict=False)

        evaluator = name_to_evaluator[commonlit_dataset_property['evaluator']]()
        for collate_batch in valid_loader:
            batch_dif = evaluator.batch_dif(model, collate_batch, commonlit_dataset_property)
            for i in range(batch_dif.size):
                output_dict['id'].append(collate_batch[f"{commonlit_dataset_property['name']}_id"][i])
                output_dict['fold'].append(fold)
                output_dict['dif'].append(batch_dif[i])

        valid_metric = evaluator.eval(model, train_loader, valid_loader, commonlit_dataset_property)
        k_fold_metrics.append(valid_metric)
        print(f"fold {fold} metric: {valid_metric}")
    
    k_fold_metrics = np.array(k_fold_metrics)
    print(k_fold_metrics)
    print(f"k fold metric: {np.mean(k_fold_metrics)} +- {np.std(k_fold_metrics)}")

    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv(config.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="code for debug")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dp", action="store_true", default=False,
                        help="use dataparallel or not")
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()
    config = load_config(args.config_path)['key']
    for key in config:
        setattr(args, key, config[key])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.mode == 'add_fold':
        add_fold(args)
    elif args.mode == 'error_score':
        analyze_error_score(args)
