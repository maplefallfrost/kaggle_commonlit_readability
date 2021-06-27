from models.roberta import Roberta
from models.self_distill import SelfDistill
from dataset import (
    CommonLitDataset,
    CommonLitSoftLabelDataset
)

eps = 1e-10

from evaluator import (
    RMSE_Evaluator
)

model_type_to_model = {
    "roberta": Roberta,
    "self-distill": SelfDistill
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}

name_to_dataset_class = {
    'commonlit': CommonLitDataset,
    'commonlit_soft_label': CommonLitSoftLabelDataset
}