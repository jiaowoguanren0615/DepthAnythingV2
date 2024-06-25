import torch
import torch.nn.functional as F


def scale_and_shift_invariant_loss(y_pred, y_true):
    """
    params:
    y_pred (torch.Tensor):
    y_true (torch.Tensor):

    return:
    torch.Tensor: scale shift invariant loss
    """

    loss = torch.abs(y_pred - y_true)

    # l2 norm
    norm = torch.norm(y_pred, p=2, dim=1, keepdim=True)

    l_ssi = torch.mean(loss / norm)

    return l_ssi



def gradient_matching_loss(y_pred, y_true):
    """
    params:
    y_pred (torch.Tensor):
    y_true (torch.Tensor):

    return:
    torch.Tensor: gradient matching loss
    """
    # compute gradiant

    y_true_onehot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()
    y_true_onehot = y_true_onehot.detach().requires_grad_()

    grad_pred = torch.autograd.grad(y_pred.sum(), y_pred, create_graph=True)[0]
    grad_true = torch.autograd.grad(y_true_onehot.sum(), y_true_onehot, create_graph=True)[0]

    # compute L1 loss
    l_gm = F.l1_loss(grad_pred, grad_true)
    return l_gm