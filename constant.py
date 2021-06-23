from models.roberta import Roberta

from evaluator import (
    RMSE_Evaluator
)

model_name_to_model = {
    "roberta-base": Roberta,
    "roberta-large": Roberta
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}