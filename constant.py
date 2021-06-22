from models.roberta import RobertaBase

from evaluator import (
    RMSE_Evaluator
)

model_name_to_model = {
    "roberta-base": RobertaBase
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}