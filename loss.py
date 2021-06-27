import torch

# use in distribution loss to bound std in the training set
std_lower_bound = 0.4

# use in torch.log to avoid nan
eps = 1e-10

class LossWrapper:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def forward(self, collate_batch, output_dict, **kwargs):
        dataset_name = next(iter(output_dict)).split("_")[0]
        if self.loss_name == 'MSE':
            label_name = "_".join([dataset_name, "mean"])
            pred_label = output_dict[label_name]
            ground_truth_label = collate_batch[label_name]
            return MSE_loss(pred_label, ground_truth_label)
        
        if self.loss_name == 'RMSE':
            label_name = "_".join([dataset_name, "mean"])
            pred_label = output_dict[label_name]
            ground_truth_label = collate_batch[label_name]
            return RMSE_loss(pred_label, ground_truth_label)
        
        if self.loss_name == 'dist':
            label_name = "_".join([dataset_name, "mean"])
            pred_mean = output_dict[label_name].squeeze()
            standard_error_name = "_".join([dataset_name, "standard_error"])
            pred_std = output_dict[standard_error_name].squeeze()
            pred_std = torch.exp(pred_std)
            gt_mean = collate_batch[label_name].squeeze()
            gt_std = collate_batch[standard_error_name].squeeze()
            gt_std[gt_std < std_lower_bound] = std_lower_bound
            return Gaussian_kl_loss(pred_mean, pred_std, gt_mean, gt_std)
        
        if self.loss_name == 'KL_div':
            label_name = "_".join([dataset_name, "soft_label"])
            pred_soft_label = output_dict[label_name]
            gt_soft_label = collate_batch[label_name]
            return KL_div_loss(pred_soft_label, gt_soft_label)
        
        if self.loss_name == 'NLL':
            label_name = "_".join([dataset_name, "soft_label"])
            pred_soft_label = output_dict[label_name]
            gt_soft_label = collate_batch[label_name]
            return NLL_loss(pred_soft_label, gt_soft_label)


def MSE_loss(pred, gt):
    loss_fn = torch.nn.MSELoss()
    pred = pred.view(-1).to(gt.dtype)
    loss = loss_fn(pred, gt.view(-1))
    return loss


def RMSE_loss(pred, gt):
    loss_fn = torch.nn.MSELoss()
    pred = pred.view(-1).to(gt.dtype)
    loss = torch.sqrt(loss_fn(pred, gt.view(-1)))
    return loss


def Gaussian_kl_loss(pred_mean, pred_std, gt_mean, gt_std):
    p = torch.distributions.Normal(pred_mean, pred_std)
    q = torch.distributions.Normal(gt_mean, gt_std)
    kl_loss = torch.distributions.kl_divergence(p, q)
    return torch.mean(kl_loss)


def Gaussian_js_loss(pred_mean, pred_std, gt_mean, gt_std):
    p = torch.distributions.Normal(pred_mean, pred_std)
    q = torch.distributions.Normal(gt_mean, gt_std)
    p_q_kl = torch.distributions.kl_divergence(p, q)
    q_p_kl = torch.distributions.kl_divergence(q, p)
    js_loss = 0.5 * (p_q_kl + q_p_kl)
    return torch.mean(js_loss)


def KL_div_loss(pred, gt):
    kl_div_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    log_gt = torch.log(gt + eps)
    kl_div_loss = kl_div_loss_fn(log_gt, pred)
    return kl_div_loss


def NLL_loss(pred, gt):
    nll_loss_fn = torch.nn.NLLLoss()
    gt_argmax = torch.argmax(gt, dim=1)
    nll_loss = nll_loss_fn(torch.log(pred + eps), gt_argmax)
    return nll_loss
