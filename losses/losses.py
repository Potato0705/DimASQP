import torch
import torch.nn.functional as F
def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred, mask_rate=None, tril_mask=False):
    """给GlobalPointer设计的交叉熵
    """
    if mask_rate is not None:
        if y_pred.is_cuda:
            # Avoid CUDA graph RNG mismatch by using the active RNG on y_pred's device.
            with torch.cuda.device(y_pred.device):
                masked = torch.rand(y_true.shape, device=y_pred.device) < (1 - mask_rate)
        else:
            masked = torch.rand(y_true.shape, device=y_pred.device) < (1 - mask_rate)
        masked = masked + y_true > 0

        if y_true.ndim == 4:
            mask_row_col = torch.any(y_true>0, dim=-3, keepdim=True)
            masked = masked | mask_row_col

        if tril_mask:
            mask_tril = torch.triu(torch.ones_like(masked), diagonal=0)
            masked = masked & mask_tril

        # y_true == -1的，mask掉
        masked = torch.where(y_true < 0, torch.zeros_like(masked), masked)

        masked = masked.to(y_pred.dtype)
        # print(masked.sum())
        y_pred = y_pred - (1 - masked) * 1e12

    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))
    # return  0.5 * (torch.mean(multilabel_categorical_crossentropy(y_true_q, y_pred_q)) + torch.mean(multilabel_categorical_crossentropy(y_true_p, y_pred_p))) + 4 * kl_loss
    return torch.sum(multilabel_categorical_crossentropy(y_true, y_pred)) / (torch.sum(y_true) + 1e-12)
    # return torch.sum(multilabel_categorical_crossentropy(y_true, y_pred))
