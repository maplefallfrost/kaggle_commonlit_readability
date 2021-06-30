import torch
import numpy as np
from torch.utils import data

from tqdm import tqdm
from sklearn.neighbors import KDTree

class KNNHelper:
    def __init__(self, model, train_loader, dataset_property):
        last_embs, scores = self._get_knn_embedding_score(model, train_loader, dataset_property)
        self.kd_tree = KDTree(last_embs)
        self.scores = scores
        self.dataset_property = dataset_property
        
    def _get_knn_embedding_score(self, model, train_loader, dataset_property):
        print("generating embedding for knn...")
        last_embs, scores = [], []
        for collate_batch in tqdm(train_loader):
            with torch.no_grad():
                output, _ = model(collate_batch, dataset_property=dataset_property)
            numpy_last_emb = output["_".join([dataset_property['name'], 'last_emb'])].cpu().numpy()
            last_embs.append(numpy_last_emb)
            scores.append(collate_batch["_".join([dataset_property['name'], "mean"])].cpu().numpy())
        last_embs = np.vstack(last_embs)
        scores = np.hstack(scores)
        return last_embs, scores
    
    def predict(self, preb_emb):
        k = self.dataset_property['knn_k']
        dists, neighbors = self.kd_tree.query(preb_emb, k=k)
        dists = dists / np.sum(dists, axis=1, keepdims=True)
        pred_scores = self.scores[neighbors]
        pred_scores = np.sum(dists * pred_scores, axis=1)
        return pred_scores
