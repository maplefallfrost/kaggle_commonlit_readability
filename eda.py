import pandas as pd
import numpy as np
import argparse
import os

from pathlib import Path
from sklearn.model_selection import KFold
from constant import data_split_type
from util import load_config


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="code for debug")
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

    if args.mode == 'add_fold':
        add_fold(config)
