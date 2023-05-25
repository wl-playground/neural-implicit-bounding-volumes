import numpy as np
from tensorflow import keras
from keras import layers

from src.model.kerasmlp.KerasMLP import KerasMLP


class BinaryClassification(KerasMLP):
    def __init__(self, optimiser, loss, hidden_layers, width, input_dimensions, output_dimensions, metrics_registry):
        super().__init__()
        self.model = keras.Sequential()
        self.metrics_registry = metrics_registry
        self.optimiser = optimiser
        self.loss = loss
        self.hidden_layers = hidden_layers
        self.width = width
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

    def build(self):
        # TODO: look at input_shape value
        self.model.add(keras.layers.Dense(units=self.input_dimensions, activation='linear', input_shape=[2]))

        for _ in range(self.hidden_layers):
            self.model.add(layers.Dense(self.width, activation="relu"))

        self.model.add(layers.Dense(self.output_dimensions, activation="sigmoid"))
        self.model.summary()

    def compile(self):
        self.model.compile(optimizer=self.optimiser, loss=self.loss, metrics=['accuracy'])

    def train(self, x_train, y_target, batch_size, epochs, verbose_mode, class_weight=None):
        verbose_mode = self.enable_verbose_output(verbose_mode)

        # model.fit returns a history of the model, currently not used
        hist = self.model.fit(
            np.array(x_train),
            np.array(y_target),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose_mode,
            validation_split=0.2,
            class_weight=class_weight
        )

    def validate(self):
        pass

    def test(self):
        pass

    # TODO: improve inference()
    def inference(self, input_value, verbose_mode=1):
        y_pred = []

        for x in range(input_value.shape[0]):
            row = []

            for y in range(input_value.shape[1]):
                value = (self.model.predict(
                    np.array(([[x, y]])),
                    verbose=self.enable_verbose_output(verbose_mode)
                ) > 0.5).astype('int32')
                row.append(value)

            y_pred.append(row)

        y_pred = np.asarray(y_pred)

        return y_pred

    # enable or disable verbose output in training and inference
    def enable_verbose_output(self, verbose_mode):
        if verbose_mode:
            return 1
        elif not verbose_mode:
            return 0
        else:
            raise TypeError("verbose_mode is a boolean parameter")

    def save_weights(self, filename='localWeights.h5'):
        super().save_weights(filename)

    def download_weights(self, filename='localWeights.h5'):
        super().download_weights(filename)
