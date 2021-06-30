import torch
import numpy as np

from tqdm import tqdm
from knn import KNNHelper


class RMSE_Evaluator:
    def __init__(self):
        pass
    
    def eval(self, model, train_loader, valid_loader, dataset_property):
        """
        model: nn.Module
            with class method model.predict
        train_loader: DataLoader. Used in knn predict.
        valid_loader: DataLoader.
        dataset_property: dict.
        """
        model.eval()

        kwargs = {}
        if 'predict_method' in dataset_property and dataset_property['predict_method'] == 'knn':
            knn_helper = KNNHelper(model, train_loader, dataset_property)
            kwargs['knn_helper'] = knn_helper

        dataset_name = dataset_property['name']
        rmse, num_sample = 0, 0
        for collate_batch in tqdm(valid_loader):
            with torch.no_grad():
                pred = model.predict(collate_batch, dataset_property, **kwargs)
            labels = collate_batch[f"{dataset_name}_mean"].cpu().numpy()
            rmse += self._eval_batch(pred, labels)
            num_sample += pred.shape[0]
        rmse = np.sqrt(rmse / num_sample)
        return rmse
    
    def _eval_batch(self, pred, labels):
        dif = pred - labels
        return np.sum(dif ** 2)
