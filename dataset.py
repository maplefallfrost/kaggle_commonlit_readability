import torch
import pandas as pd

from torch.nn.utils.rnn import pad_sequence


class Collator:
    """
    class for batch collate
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        Input
        batch: list[dict]. 
            List of dict by __getitem__ function of dataset class.
        Output:
        collate_batch: dict. 
            each value in dict contain object that can be processed by pytorch.
        """
        keys = batch[0].keys()
        collate_batch = dict()
        for key in keys:
            if key.find("token_ids") != -1:
                token_ids = [torch.LongTensor(x[key]) for x in batch]
                collate_data = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_token_id)
                lens = torch.LongTensor([token_id.size(0) for token_id in token_ids])
                max_len = torch.max(lens)
                attention_mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
                collate_batch['attention_mask'] = attention_mask
            else:
                collate_data = [x[key] for x in batch]
                collate_data = torch.Tensor(collate_data)
            collate_batch[key] = collate_data
        
        return collate_batch


class CommonLitDataset:
    def __init__(self, dict_data, dataset_name, subset_index=None):
        """
        dict_data: dict. 
            generated from df_to_dict function in util.py
        subset_index: list[int]. 
            subset of used data.
        """
        self.dict_data = dict_data
        self.subset_index = subset_index
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        index = self.subset_index[index]

        sample = {
            f'{self.dataset_name}_token_ids': self.dict_data['text'][index],
            f'{self.dataset_name}_label': self.dict_data['target'][index],
            f'{self.dataset_name}_standard_error': self.dict_data['standard_error'][index]
        }
        return sample
    
    def __len__(self):
        return len(self.subset_index)
