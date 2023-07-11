import jax.numpy as jnp

from src.data.ImageDataLoader import ImageDataLoader


def target(file_path):
    bunny = jnp.array(ImageDataLoader.normalise_train(ImageDataLoader.load_binary_image(file_path)), dtype="float32")

    return bunny
