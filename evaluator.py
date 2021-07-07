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

        rmse, num_sample = 0, 0
        for collate_batch in tqdm(valid_loader):
            batch_dif = self.batch_dif(model, collate_batch, dataset_property, **kwargs)
            rmse += np.sum(batch_dif ** 2)
            num_sample += batch_dif.size
        rmse = np.sqrt(rmse / num_sample)
        return rmse
    
    def batch_dif(self, model, collate_batch, dataset_property, **kwargs):
        with torch.no_grad():
            pred = model.predict(collate_batch, dataset_property, **kwargs)
        labels = collate_batch[f"{dataset_property['name']}_mean"].cpu().numpy()
        return pred - labels
