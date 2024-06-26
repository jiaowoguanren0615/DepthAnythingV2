import torch.nn.functional as F
import torch

def scale_and_shift_invariant_loss(y_pred, y_true):
    """
    scale and shift invariant loss

    params:
    y_pred (torch.Tensor):
    y_true (torch.LongTensor):

    return:
    torch.Tensor: scale and shift invariant loss
    """
    # convert to one-hot
    y_true_onehot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

    # [B, H, W, num_classes] -----> [B, num_classes, H, W]
    y_true_onehot = y_true_onehot.transpose(1, 3)
    y_pred = y_pred.float()

    # compute mean & std
    y_pred_mean = y_pred.mean(dim=1, keepdim=True)
    y_pred_std = y_pred.std(dim=1, keepdim=True)
    y_true_mean = y_true_onehot.mean(dim=1, keepdim=True)
    y_true_std = y_true_onehot.std(dim=1, keepdim=True)

    # compute scale and shift invariant loss
    loss = F.mse_loss((y_pred - y_pred_mean) / (y_pred_std + 1e-8),
                      (y_true_onehot - y_true_mean) / (y_true_std + 1e-8))

    return loss



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

    # [B, H, W, num_classes] -----> [B, num_classes, H, W]
    y_true_onehot = y_true_onehot.transpose(1, 3)
    y_true_onehot = y_true_onehot.detach().requires_grad_()

    # y_pred = y_pred.float().requires_grad_()
    y_pred = y_pred.float().requires_grad_()

    # compute gradiant
    grad_pred = torch.autograd.grad(y_pred.sum(), y_pred, create_graph=True)[0]
    grad_true = torch.autograd.grad(y_true_onehot.sum(), y_true_onehot, create_graph=True)[0]

    # compute l1_loss
    l_gm = F.l1_loss(grad_pred, grad_true)
    return l_gm

#
# if __name__ == '__main__':
#     y_true = torch.ones(4, 518, 518).long()
#     y_pred = torch.ones(4, 19, 518, 518).long()
#     ssi = scale_and_shift_invariant_loss(y_pred, y_true)
#     gm = gradient_matching_loss(y_pred, y_true)
#     loss_total = ssi + gm
#     print(loss_total)