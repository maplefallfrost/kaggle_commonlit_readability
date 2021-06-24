import torch
import numpy as np

from util import to_device
from tqdm import tqdm

class RMSE_Evaluator:
    def __init__(self, device):
        self.device = device
    
    def eval(self, model, loader, dataset_property):
        """
        model: nn.Module
            with class method model.predict
        loader: DataLoader.
        dataset_property: dict.
        """
        model.eval()
        dataset_name = dataset_property['name']
        rmse, num_sample = 0, 0
        for collate_batch in tqdm(loader):
            to_device(collate_batch, self.device)
            with torch.no_grad():
                pred = model.predict(collate_batch, dataset_property)
            labels = collate_batch[f"{dataset_name}_label"]
            pred = pred.view(-1).to(labels.dtype).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            rmse += self._eval_batch(pred, labels)
            num_sample += pred.shape[0]
        rmse = np.sqrt(rmse / num_sample)
        model.train()
        return rmse
    
    def _eval_batch(self, pred, labels):
        dif = pred - labels
        return np.sum(dif ** 2)
