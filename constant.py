from models.huggingface import MLMModel
from models.ensemble import EnsembleModel
from dataset import (
    CommonLitDataset,
    CommonLitSoftLabelDataset
)

from evaluator import (
    RMSE_Evaluator
)

model_type_to_model = {
    "roberta": MLMModel,
    "bert": MLMModel,
    "self-distill": EnsembleModel
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}

name_to_dataset_class = {
    'commonlit': CommonLitDataset,
    'commonlit_soft_label': CommonLitSoftLabelDataset
}

data_split_type = {'train': 0, 'valid': 1, 'train_extra': 2, 'no_use': 3}