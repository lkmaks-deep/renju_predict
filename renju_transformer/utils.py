import torch
from torch import nn


def transform_to_tokens(lst, W=15):
    res = []
    for tup in lst:
        res.append(tup[0] * W + tup[1])
    return res


def pad_tokens(arrs_list, pad_token_id=15 * 15):
    max_len = max(len(lst) for lst in arrs_list)
    res = []
    for arr in arrs_list:
        res.append(arr + [pad_token_id] * (max_len - len(arr)))
    return res


def PerplexityLoss(logits, true_tokens, pad_token_id=15 * 15):
    """
    Args:
        logits: [seq_len, batch_size, vocab_size]
        true_tokens: [batch_size, seq_len]

    Returns: loss

    """
    true_tokens = true_tokens.transpose(0, 1)
    mask = true_tokens != pad_token_id
    flat_logits = logits[:-1, :, :].view(-1, logits.size(-1))
    flat_tokens = true_tokens.reshape(-1)
    flat_mask = mask.reshape(-1)
    log_probs = nn.functional.log_softmax(flat_logits, dim=-1) * flat_mask.view(-1, 1)
    return torch.mean(-log_probs[torch.arange(flat_tokens.shape[0]), flat_tokens])
