import torch
from torch.nn import functional as F


@torch.jit.script
def cross_entropy(pred, target):
    # Both branches receive gradients, matching the research code.
    return -(target.softmax(1) * pred.log_softmax(1)).sum(1)


@torch.jit.script
def symmetric_consistency(p, q):
    loss1 = cross_entropy(p, q)
    loss2 = cross_entropy(q, p) + 2 * cross_entropy(q, q)
    return 0.5 * (loss1 + loss2)


def consistency_loss(logits, batch_size):
    p, q = logits[batch_size : 2 * batch_size], logits[2 * batch_size :]
    return symmetric_consistency(p, q).mean(0)


def alignment_loss(records, source):
    loss = records[0][0].new_zeros(())
    for current, reference in zip(records, source):
        source_mean, source_var = reference[0].squeeze(), reference[1].squeeze()
        mean, variance = current[0].squeeze(), current[1].squeeze()
        # Keep the evaluation objective's GB-KL scale and summation order.
        term = (
            -variance.log() + variance / source_var + (mean - source_mean).square() / source_var
        ).mean()
        term = source_var.sqrt().mean() * term
        loss = loss + term
    return loss


def correlation_loss(adapters, config):
    loss = adapters[0].weight.new_zeros(())
    for layer in adapters:
        rank, dim = layer.r, layer.in_features
        factors = [
            layer.lora_A[:rank],
            layer.lora_A[rank:],
            layer.lora_B[:dim].T,
            layer.lora_B[dim:].T,
        ]
        for factor in factors:
            gram = factor @ factor.T
            target = (
                config.gram_target
                * factor.shape[1]
                / dim
                * torch.eye(rank, device=factor.device, dtype=factor.dtype)
            )
            loss = loss + F.mse_loss(gram, target, reduction="mean")
    return loss
