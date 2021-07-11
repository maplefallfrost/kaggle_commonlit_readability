import argparse
import yaml
import re
import torch
import numpy as np

from collections import OrderedDict

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


def df_to_dict(df, tokenizer, text_column):
    """
    Input
    df: pd.DataFrame. original csv
    tokenizer: tokenizer in HuggingFace.
    text_column: str. name of text column

    Output
    data: dict. 
        with keys in df, where text column is parsed by tokenizer 
        'token_ids': list[list[int]]
    """
    data = dict()
    for key in df.columns:
        if key != text_column:
            data[key] = df[key].tolist()
        else:
            all_token_ids = []
            for raw_text in df[key].tolist():
                raw_text = raw_text.replace('\n', ' ')
                token_ids = tokenizer(raw_text)['input_ids']
                all_token_ids.append(token_ids)
            data[key] = all_token_ids

    return data


def to_device(collate_batch, device):
    device_batch = dict()
    for key in collate_batch:
        if isinstance(collate_batch[key], torch.Tensor):
            # device_batch[key] = collate_batch[key].to(device, non_blocking=True)
            device_batch[key] = collate_batch[key].to(device)
        else:
            device_batch[key] = collate_batch[key]
    return device_batch
    

def load_config(config_path):
    """
    Input
    config_path: Path. Path of yaml file.
    Output
    config: argparse.Namespace
    """

    # This script is to solve the bug of loading scientific notation(e.g. 5e-5) as str in yaml
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_path, "r") as fp:
        config = yaml.load(fp, Loader=loader)
    return config


def load_state_dict(model_save_path):
    state_dict = torch.load(model_save_path)
    key = next(iter(state_dict.keys()))
    if key.find("module") != -1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    return state_dict


def extract_dataset_name(output_dict):
    dataset_name = next(iter(output_dict)).split("_")[0]
    return dataset_name


def get_class_to_score(range_min, range_max, interval):
    score_range = np.arange(range_min, range_max + interval, interval)
    class_to_score = np.zeros(shape=(1, len(score_range) - 1))
    for i in range(len(score_range) - 1):
        class_to_score[0][i] = (score_range[i] + score_range[i + 1]) / 2
    return torch.Tensor(class_to_score)


def prob_to_mean(probs, class_to_score):
    if class_to_score.size(1) != probs.size(1):
        raise ValueError("class_to_score.size(1) should be equal to probs.size(1). Please check range_min, range_max, and interval.")
    batch_class_to_score = class_to_score.expand_as(probs)
    batch_mean = torch.sum(batch_class_to_score * probs, axis=1)
    return batch_mean
