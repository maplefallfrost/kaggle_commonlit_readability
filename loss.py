import torch

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