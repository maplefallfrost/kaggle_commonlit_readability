import torch

# use in distribution loss to bound std in the training set
std_lower_bound = 0.4

class LossWrapper:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def forward(self, collate_batch, output_dict, **kwargs):
        dataset_name = next(iter(output_dict)).split("_")[0]
        label_name = "_".join([dataset_name, "label"])
        if self.loss_name == 'MSE':
            pred_label = output_dict[label_name]
            ground_truth_label = collate_batch[label_name]
            return MSELoss(pred_label, ground_truth_label)
        
        if self.loss_name == 'RMSE':
            pred_label = output_dict[label_name]
            ground_truth_label = collate_batch[label_name]
            return RMSELoss(pred_label, ground_truth_label)
        
        if self.loss_name == 'dist':
            pred_mean = output_dict[label_name].squeeze()
            standard_error_name = "_".join([dataset_name, "standard_error"])
            pred_std = output_dict[standard_error_name].squeeze()
            pred_std = torch.exp(pred_std)
            ground_truth_mean = collate_batch[label_name].squeeze()
            ground_truth_std = collate_batch[standard_error_name].squeeze()
            ground_truth_std[ground_truth_std < std_lower_bound] = std_lower_bound
            p = torch.distributions.Normal(pred_mean, ground_truth_std)
            q = torch.distributions.Normal(ground_truth_mean, ground_truth_std)
            dist_loss = torch.distributions.kl_divergence(p, q)
            return torch.mean(dist_loss)


def MSELoss(logits, labels):
    loss_fn = torch.nn.MSELoss()
    logits = logits.view(-1).to(labels.dtype)
    loss = loss_fn(logits, labels.view(-1))
    return loss


def RMSELoss(logits, labels):
    loss_fn = torch.nn.MSELoss()
    logits = logits.view(-1).to(labels.dtype)
    loss = torch.sqrt(loss_fn(logits, labels.view(-1)))
    return loss