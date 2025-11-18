import torch
import torch.nn.functional as F
from einops import rearrange


def compute_logprobs(logits, target):

    log_probs = F.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=target.unsqueeze(-1)
    ).squeeze(-1)

    return selected_log_probs


def compute_cross_entropy_loss(logits, target, ignore_index):

    loss = torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index,
            reduction = "none"
    )
    loss = loss.mean(-1).mean()

    return loss