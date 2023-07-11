import jax.numpy as jnp
import jax
from jax import grad

from src.model.neuralnetwork.jaxmlp.loss import bce_loss
from src.model.neuralnetwork.jaxmlp.model import model
from src.model.neuralnetwork.jaxmlp.target import target
from src.view.RendererFactory import get_renderer, ImageVisualisationType

x = jnp.arange(-32, 32, 1)
y = jnp.arange(-32, 32, 1)
X, Y = jnp.meshgrid(x, y, indexing="xy")
print("training image size", X.shape, "\n\n")

theta = jnp.array([
    0.5, 0.0, 0.0,
    -0.5, 0.0, 0.0,
    0.0, 0.5, 5.0,
    0.0, 1.0, 0.0
])

y_target = target('/content/neural-implicit-bounding-volumes/data/2D/new_target_64x64.png')

# autograd of loss function
grad_loss = jax.jit(grad(bce_loss))

# JIT compile functions
jit_loss = jax.jit(bce_loss)
jit_model = jax.jit(model)

# define class weights
positive_class_weight = 1.0
negative_class_weight = 1.0 / 35.0

# learning rate
learning_rate = 0.01


renderer = get_renderer(ImageVisualisationType.COMPARISONWITHOVERLAY)

# visualise decision boundaries at initialisation
renderer.render(y_target, jnp.where(jit_model(theta, X, Y) >= 0.5, 1.0, 0.0))
print("init\n")

# training loop
for iteration in range(5000000):
    theta -= learning_rate * grad_loss(
        theta,
        X=X,
        Y=Y,
        target=y_target,
        positive_class_weight=positive_class_weight,
        negative_class_weight=negative_class_weight
    )

    if (iteration + 1) % 50000 == 0:
        renderer.render(y_target, jnp.where(jit_model(theta, X, Y) >= 0.5, 1.0, 0.0))

        loss = jit_loss(
            theta,
            X=X,
            Y=Y,
            target=y_target,
            positive_class_weight=positive_class_weight,
            negative_class_weight=negative_class_weight
        )

        print("iteration {}, loss {}\n".format(iteration + 1, loss))

# final theta after training
print("theta", theta)
