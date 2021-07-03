from models.huggingface import MLMModel
from models.self_distill import SelfDistill
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
    "self-distill": SelfDistill
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}

name_to_dataset_class = {
    'commonlit': CommonLitDataset,
    'commonlit_soft_label': CommonLitSoftLabelDataset
}