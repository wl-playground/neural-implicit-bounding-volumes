import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def bce_loss(
        out,
        target,
        positive_class_weight=1.0,
        negative_class_weight=1.0
):
    epsilon = 1e-7

    positive_class = target * torch.maximum(
        positive_class_weight * torch.log(out + epsilon),  # add epsilon for numerical stability
        torch.full(out.shape, -100.0).to(device)
    )  # clamp maximum loss to not introduce infinite values

    negative_class = (1.0 - target) * torch.maximum(
        negative_class_weight * torch.log((1.0 - out) + epsilon),  # add epsilon for numerical stability
        torch.full(out.shape, -100.0).to(device)
    )  # clamp maximum loss to not introduce infinite values

    loss = -torch.mean(positive_class + negative_class)

    return loss
