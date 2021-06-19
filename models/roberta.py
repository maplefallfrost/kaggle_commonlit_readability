from logging import raiseExceptions
import torch
from torch._C import Value
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM
from models.util import create_last_layers
from modules.weighted_layer_pooling import WeightedLayerPooling

class RobertaBase(nn.Module):
    def __init__(self, config):
        """
        config: argparse.Namespace.
        """
        super().__init__()
        self.embedding_method = config.embedding_method
        dataset_properties = config.dataset_properties
        model_name = "roberta-base"
        model_config = AutoConfig.from_pretrained(model_name)
        self.roberta_base = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
        last_layers = create_last_layers(dataset_properties, model_config.hidden_size)
        self.last_layers = nn.ModuleDict({name: layer for name, layer in last_layers.items()})

        if self.embedding_method == 'weight-pool':
            self.output_emb_layer = WeightedLayerPooling(
                model_config.num_hidden_layers,
                layer_start=config.embedding_layer_start,
                layer_weights=None)
    
    def _get_last_embedding(self, token_ids, **kwargs):
        attention_mask = kwargs.get('attention_mask', None)
        output_dict = self.roberta_base(token_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            return_dict=True)
        hidden_states = output_dict['hidden_states']
        if self.embedding_method == 'last-avg':
            last_hidden_states = hidden_states[-1]
            avg_last_hidden_states = torch.mean(last_hidden_states, dim=1)
            return avg_last_hidden_states
        elif self.embedding_method == 'weight-pool':
            last_emb = self.output_emb_layer(hidden_states)
            return last_emb
        else:
            raise ValueError(f"Unsupported embedding_method {self.embedding_method}.")


    def forward(self, collate_batch):
        """
        Input
        collate_batch: dict. generated by correspoding Collator.
            assume key '{dataset_name}_token_ids' exists.
        Ouptut
        output: torch.Tensor. size [batch_size x num_classes]
        """
        token_ids = None
        for key in collate_batch.keys():
            if key.find("token_ids") != -1:
                dataset_name = key.split("_")[0]
                token_ids = collate_batch[key]
        if token_ids is None:
            raise ValueError("collate_batch should contain key '\{dataset_name\}_token_ids'")

        output = dict()
        last_embedding = self._get_last_embedding(token_ids, **collate_batch)
        try:
            layer = self.last_layers[dataset_name]
        except Exception:
            raise ValueError(f"{dataset_name} last layer doesn't exist. Please check your Collator implementation.")
            
        output = layer(last_embedding)
        return output
    
    def predict(self, collate_batch, dataset_property):
        """
        Input
        collate_batch: dict. generated by correspoding Collator.
            assume key '{dataset_name}_token_ids' exists.
        Ouptut
        output: torch.Tensor. size [batch_size]
        """
        output = self.forward(collate_batch)
        task = dataset_property["task"]
        if task == 'reg':
            return output
        elif task == 'cls':
            pred_label = torch.argmax(output, dim=-1)
            return pred_label
        else:
            raise ValueError(f"Unknown task {task}. Should be in (cls/reg)")

