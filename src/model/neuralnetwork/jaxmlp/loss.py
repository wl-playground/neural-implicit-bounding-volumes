# implementation as per PyTorch

from src.model.neuralnetwork.jaxmlp.model import model, model_aabb, model_polygon
import jax
import jax.numpy as jnp


# for square 3D hut
def bce_loss(
        theta,
        X,
        Y,
        target,
        positive_class_weight=1.0,
        negative_class_weight=1.0
):
    epsilon = 1e-7

    out = model(theta, X, Y)

    positive_class = target * jax.lax.max(
        positive_class_weight * jax.lax.log(out + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    negative_class = (1.0 - target) * jax.lax.max(
        negative_class_weight * jax.lax.log((1.0 - out) + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    loss = -jnp.mean(positive_class + negative_class)

    return loss


# for circle 3D hut
def bce_loss_polygon(
        theta,
        X,
        Y,
        target,
        positive_class_weight=1.0,
        negative_class_weight=1.0
):
    epsilon = 1e-7

    out = model_polygon(theta, X, Y)

    positive_class = target * jax.lax.max(
        positive_class_weight * jax.lax.log(out + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    negative_class = (1 - target) * jax.lax.max(
        negative_class_weight * jax.lax.log((1.0 - out) + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    loss = -jnp.mean(positive_class + negative_class)

    return loss


# for 3 plane square 3D hut
def bce_loss_aabb(
        theta,
        X,
        Y,
        target,
        positive_class_weight=1.0,
        negative_class_weight=1.0
):
    epsilon = 1e-7

    out = model_aabb(theta, X, Y)

    positive_class = target * jax.lax.max(
        positive_class_weight * jax.lax.log(out + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    negative_class = (1.0 - target) * jax.lax.max(
        negative_class_weight * jax.lax.log((1.0 - out) + epsilon),  # add epsilon for numerical stability
        jnp.full(out.shape, -100.0)
    )  # clamp maximum loss to not introduce infinite values, as per PyTorch

    loss = -jnp.mean(positive_class + negative_class)

    return loss
