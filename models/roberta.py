import torch
import torch.nn as nn
import os

from transformers import AutoConfig, AutoModelForMaskedLM
from models.util import create_last_layers
from modules.weighted_layer_pooling import WeightedLayerPooling
from models.huggingface import HuggingFaceBaseModel

class Roberta(HuggingFaceBaseModel):
    def __init__(self, config):
        """
        config: argparse.Namespace.
        """
        super().__init__()

        self.embedding_method = config.embedding_method
        dataset_properties = config.dataset_properties

        # if load from mlm pretrained model
        if hasattr(config, "pretrained_dir") and config.pretrained_dir:
            if not os.path.exists(config.pretrained_dir):
                raise ValueError(f"pretrained dir {config.pretrained_dir} not exist")

            # print(f"load from pretrained model from {config.pretrained_dir}")
            model_config_path = os.path.join(config.pretrained_dir, "config.json")
            model_config = AutoConfig.from_pretrained(model_config_path)
            if hasattr(config, "model_config_dict"):
                for key in config.model_config_dict:
                    model_config.__dict__[key] = config.model_config_dict[key]
            self.backbone = AutoModelForMaskedLM.from_config(model_config)
            pretrained_model_path = os.path.join(config.pretrained_dir, "pytorch_model.bin")
            pretrained_model = torch.load(pretrained_model_path)
            self.backbone.load_state_dict(pretrained_model)
        else:
            model_config = AutoConfig.from_pretrained(config.model_name)
            if hasattr(config, "model_config_dict"):
                for key in config.model_config_dict:
                    model_config.__dict__[key] = config.model_config_dict[key]
            self.backbone = AutoModelForMaskedLM.from_pretrained(config.model_name, config=model_config)
        
        last_layers = create_last_layers(dataset_properties, model_config.hidden_size)
        self.last_layers = nn.ModuleDict({name: layer for name, layer in last_layers.items()})

        if self.embedding_method == 'weight-pool':
            self.output_emb_layer = WeightedLayerPooling(
                model_config.num_hidden_layers,
                layer_start=config.embedding_layer_start,
                layer_weights=None)
    