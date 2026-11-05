"""Optional recovery-only MLX loss hook; never patches an installed package.

Install in a *new* training invocation with trainer.train(..., loss=loss).
The tokenizer gate records and probes this exact implementation when selected.
"""


def loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    positions = mx.arange(1, targets.shape[1] + 1)
    # lengths[:,0] is the first final-assistant token in the original sequence.
    # lengths[:,1] is the exclusive end of real tokens, not a padding token.
    mask = mx.logical_and(positions >= lengths[:, 0:1], positions < lengths[:, 1:])
    count = mask.sum()
    token_losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
    return (token_losses * mask).sum() / count, count
