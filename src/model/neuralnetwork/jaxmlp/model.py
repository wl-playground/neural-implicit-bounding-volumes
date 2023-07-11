from src.model.neuralnetwork.jaxmlp.activation import relu
import jax


# model defined for binary classification
def model(theta, X, Y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3, w41, w42, b4 = theta

    ramp1 = relu(w11*X + w12*Y + b1)
    ramp2 = relu(w21*X + w22*Y + b2)
    ramp3 = (w31*X + w32*Y + b3)
    ramp4 = relu(w41*X + w42*Y + b4)

    ramp5 = jax.nn.sigmoid(ramp3 - ramp1 - ramp2 - ramp4)

    return ramp5


# model defined for regression
def model_square_relu(theta, X, Y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3, w41, w42, b4 = theta

    ramp1 = relu(w11*X + w12*Y + b1)
    ramp2 = relu(w21*X + w22*Y + b2)
    ramp3 = relu(w31*X + w32*Y + b3)
    ramp4 = relu(w41*X + w42*Y + b4)

    ramp5 = ramp3 - ramp1 - ramp2 - ramp4

    return ramp5


# model of square 3d hut defined using 3 relu planes
def model_aabb(theta, X, Y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3 = theta

    ramp1 = relu(w11*X + w12*Y + b1)
    ramp2 = relu(w21*X + w22*Y + b2)
    ramp3 = relu(w31*X + w32*Y + b3)

    aabb = jax.nn.sigmoid(relu(ramp1 - ramp2 - ramp3) - 0.55)

    return aabb


# model defined for binary classification
def model_polygon(theta, X, Y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3, w41, w42, b4, w51, w52, b5, w61, w62, b6, w71, w72, b7 = theta

    ramp1 = relu(w11*X + w12*Y + b1)
    ramp2 = relu(w21*X + w22*Y + b2)
    ramp3 = relu(w31*X + w32*Y + b3)
    ramp4 = relu(w41*X + w42*Y + b4)
    ramp5 = relu(w51*X + w52*Y + b5)
    ramp6 = relu(w61*X + w62*Y + b6)
    ramp7 = relu(w71*X + w72*Y + b7)

    polygon = jax.nn.sigmoid(ramp1 - ramp2 - ramp3 - ramp4 - ramp5 - ramp6 - ramp7)

    return polygon


# model defined for regression

def model_polygon_relu(theta, X, Y):
    w11, w12, b1, w21, w22, b2, w31, w32, b3, w41, w42, b4, w51, w52, b5, w61, w62, b6, w71, w72, b7 = theta

    ramp1 = relu(w11*X + w12*Y + b1)
    ramp2 = relu(w21*X + w22*Y + b2)
    ramp3 = relu(w31*X + w32*Y + b3)
    ramp4 = relu(w41*X + w42*Y + b4)
    ramp5 = relu(w51*X + w52*Y + b5)
    ramp6 = relu(w61*X + w62*Y + b6)
    ramp7 = relu(w71*X + w72*Y + b7)

    polygon = relu(ramp1 - ramp2 - ramp3 - ramp4 - ramp5 - ramp6 - ramp7)

    return polygon
