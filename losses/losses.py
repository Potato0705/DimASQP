import torch

INF = 1e12


def multilabel_categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    GlobalPointer 标准多标签交叉熵（logsumexp 版）
    - y_pred: logits（不要先 sigmoid）
    - y_true: {0,1}
    这个版本对“极度稀疏正样本”的 LxL 矩阵非常关键，能避免模型学成全0。
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    # (1-2y)*logits：正样本取反，统一写法
    y_pred = (1.0 - 2.0 * y_true) * y_pred

    # 正样本只看 pos_loss；负样本只看 neg_loss
    y_pred_neg = y_pred - y_true * INF          # 把正样本位置变成 -INF，只剩负样本竞争
    y_pred_pos = y_pred - (1.0 - y_true) * INF  # 把负样本位置变成 -INF，只剩正样本竞争

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    attention_mask: torch.Tensor = None,
    tril_mask: bool = False
) -> torch.Tensor:
    """
    GlobalPointer 总损失（支持 padding mask / 下三角 mask）
    - y_pred: logits
    - y_true: {0,1} 或含 -1（无效位置）
    """
    # 允许 y_true 里有 -1 表示无效位置
    valid = (y_true >= 0)
    y_true = torch.clamp(y_true, 0, 1).float()

    B, H, L, _ = y_pred.shape
    device = y_pred.device

    # 先基于 attention_mask 构造合法 (i,j) 区域
    if attention_mask is not None:
        am = attention_mask.bool()  # [B,L]
        m = am[:, None, :, None] & am[:, None, None, :]  # [B,1,L,L]
    else:
        m = torch.ones((B, 1, L, L), dtype=torch.bool, device=device)

    if tril_mask:
        tri = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=0)  # 上三角含对角
        m = m & tri[None, None, :, :]

    # 合并 y_true 的 valid mask（-1 区域）
    m = m & valid.bool()

    # 无效区域：logits 设 -INF，label 设 0
    y_pred = y_pred.masked_fill(~m, -INF)
    y_true = y_true.masked_fill(~m, 0.0)

    # reshape: (B*H, L*L)
    y_pred = y_pred.reshape(B * H, -1)
    y_true = y_true.reshape(B * H, -1)

    loss = multilabel_categorical_crossentropy(y_true, y_pred)  # [B*H]
    return loss.mean()
