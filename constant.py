from models.roberta import Roberta
from models.self_distill import SelfDistill

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