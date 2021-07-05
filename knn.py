import torch
import numpy as np

from tqdm import tqdm, trange
from sklearn.neighbors import KDTree

class KNNHelper:
    def __init__(self, model, train_loader, dataset_property):
        last_embs, scores = self._get_knn_embedding_score(model, train_loader, dataset_property)
        self.kd_tree = KDTree(last_embs)
        self.last_embs = last_embs
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
    
    def predict(self, pred_emb, **kwargs):
        k = self.dataset_property['knn_k']
        del_index = kwargs.get('del_index', None)
        if del_index is not None:
            k += 1

        dists, neighbors = self.kd_tree.query(pred_emb, k=k)
        pred_scores = self.scores[neighbors]

        extra_emb = kwargs.get('extra_emb', None)
        if extra_emb is not None:
            extra_score = kwargs['extra_score']
            extra_dist = np.linalg.norm(extra_emb - pred_emb)
            argmax_dists = np.argmax(dists, axis=1)
            for i, d in enumerate(argmax_dists):
                if d > extra_dist:
                    dists[i, d] = extra_dist
                    pred_scores[i, d] = extra_score
        
        if del_index is not None:
            rows, cols = np.where(neighbors == del_index)
            for row, col in zip(rows, cols):
                dists[row, col] = dists[row, -1]
                pred_scores[row, col] = dists[row, -1]
            np.delete(dists, -1, 1)
            np.delete(pred_scores, -1, 1)
            
        dists = dists / np.sum(dists, axis=1, keepdims=True)
        pred_scores = np.sum(dists * pred_scores, axis=1)
        return pred_scores
    
    def is_text_confident(self, new_text_emb, new_text_score):
        k = self.dataset_property['knn_k']
        dists, neighbors = self.kd_tree.query(new_text_emb, k=k)
        neighbors = neighbors[0]
        embs = self.last_embs[neighbors]
        gt_scores = self.scores[neighbors]
        old_pred_scores = self.predict(embs)
        new_pred_scores = self.predict(embs, extra_emb=new_text_emb, extra_score=new_text_score)
        old_diff = np.linalg.norm(gt_scores - old_pred_scores)
        new_diff = np.linalg.norm(gt_scores - new_pred_scores)
        return new_diff < old_diff
    
    def is_noise(self, index, k):
        _, neighbors = self.kd_tree.query(self.last_embs[index].reshape(1, -1), k=k)
        neighbors = neighbors[0]
        embs = self.last_embs[neighbors]
        gt_scores = self.scores[neighbors]
        old_pred_scores = self.predict(embs)
        new_pred_scores = self.predict(embs, del_index=index)
        old_diff = np.linalg.norm(gt_scores - old_pred_scores)
        new_diff = np.linalg.norm(gt_scores - new_pred_scores)
        return new_diff < old_diff
    
    def find_noise_samples(self, k):
        noise_samples = []
        for index in trange(self.last_embs.shape[0]):
            if self.is_noise(index, k):
                noise_samples.append(index)
        return noise_samples
