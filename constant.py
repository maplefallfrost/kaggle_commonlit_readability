from models.roberta import RobertaBase
from loss import (
    MSELoss,
    RMSELoss
)
from evaluator import (
    RMSE_Evaluator
)

model_name_to_model = {
    "roberta-base": RobertaBase
}

name_to_loss = {
    'MSE': MSELoss,
    'RMSE': RMSELoss
}

name_to_evaluator = {
    'RMSE': RMSE_Evaluator
}