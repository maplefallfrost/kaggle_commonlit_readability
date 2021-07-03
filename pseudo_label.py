import argparse
from enum import unique
import os
import sys
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from util import load_config, load_state_dict, df_to_dict
from transformers import AutoTokenizer
from tqdm import tqdm
from knn import KNNHelper
from constant import model_type_to_model
from models.self_distill import SelfDistill
from dataset import CommonLitDataset, Collator
from data_loader import DataLoaderX
from collections import defaultdict


def get_tokens_id(config, tokenizer, text):
    if config.get_seq_method == 'vanilla':
        token_ids = tokenizer(text)['input_ids']
        if len(token_ids) >= config.seq_min_len and len(token_ids) <= config.seq_max_len:
            return token_ids


def update_new_sample(sample_dict, text, target, standard_error, fold, k_fold):
    sample_dict['text'].append(text)
    sample_dict['target'].append(target)
    sample_dict['standard_error'].append(standard_error)
    for k in range(k_fold):
        sample_dict[f"fold{k}"].append(int(k != fold))


def pseudo_label(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config_json_path = os.path.join(config.checkpoint_dir, "config.json")
    if not os.path.exists(config_json_path):
        os.rename(os.path.join(config.checkpoint_dir, "tokenizer_config.json"), config_json_path)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)

    pl_dataset = load_dataset(config.pl_dataset_name)

    commonlit_dataset_property = config.dataset_properties[0]

    df = pd.read_csv(commonlit_dataset_property["train_data_path"])
    dict_data = df_to_dict(df, tokenizer, text_column=commonlit_dataset_property["text_column"])
    collator = Collator(tokenizer.pad_token_id)
    dataset_name = commonlit_dataset_property['name']

    new_sample_dict = defaultdict(lambda: [])
    for fold in range(config.k_fold):
        model = model_type_to_model[config.model_type](config)
        model.eval()
        if not isinstance(model, SelfDistill):
            model = model.to(device)

        model_save_path = os.path.join(config.checkpoint_dir, f"model_{fold}.th")
        trained_state_dict = load_state_dict(model_save_path)
        model.load_state_dict(trained_state_dict)

        train_index = df[df[f"fold{fold}"] < 2].index.tolist()
        train_dataset = CommonLitDataset(dict_data=dict_data, 
            dataset_name=commonlit_dataset_property["name"],
            subset_index=train_index)

        train_loader = DataLoaderX(
            device,
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=Collator(tokenizer.pad_token_id),
            pin_memory=True,
            num_workers=3)
        
        knn_helper = KNNHelper(model, train_loader, commonlit_dataset_property)

        count, conf_count = 0, 0
        unique_text = set()

        for split_name in ['train', 'validation']:
            cur_dataset = pl_dataset[split_name]
            for text in tqdm(cur_dataset[config.pl_text_column]):
                if text in unique_text:
                    continue
                unique_text.add(text)
                token_ids = get_tokens_id(config, tokenizer, text)
                if token_ids:
                    count += 1
                    collate_batch = collator([{"_".join([dataset_name, "token_ids"]): token_ids}])
                    with torch.no_grad():
                        output_dict, _ = model(collate_batch, dataset_property=commonlit_dataset_property)
                        last_emb = output_dict["_".join([dataset_name, "last_emb"])].cpu().numpy()
                        new_text_score = output_dict["_".join([dataset_name, "mean"])].item()
                    is_confident = knn_helper.is_text_confident(last_emb, new_text_score)
                    conf_count += is_confident
                    if is_confident:
                        new_text_std = output_dict["_".join([dataset_name, "standard_error"])].item()
                        new_text_std = np.exp(new_text_std)
                        update_new_sample(new_sample_dict, text, new_text_score, new_text_std, fold, config.k_fold)
            
        print(f"count: {count}. conf count: {conf_count}")

    new_sample_df = pd.DataFrame.from_dict(new_sample_dict)
    merge_df = pd.concat([df, new_sample_df], axis=0, ignore_index=True)
    merge_df.to_csv(config.csv_save_path, index=False)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pseudo label")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--pl_config_path", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    model_config = load_config(args.config_path)['key']
    pl_config = load_config(args.pl_config_path)
    config = argparse.Namespace(**{**model_config, **pl_config})
    
    config.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pseudo_label(config)