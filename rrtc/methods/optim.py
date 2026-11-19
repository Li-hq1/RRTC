import torch


def make_optimizer(params, config):
    """Use Nesterov SGD with the evaluation code's batch-size scaling."""
    lr = config.lr / 64 * config.batch_size
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay,
    )
