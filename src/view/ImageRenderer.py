from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import numpy as np


class ImageRenderer(ABC):
    @abstractmethod
    def render(self, original, reconstruction, vmin, vmax):
        pass


class ComparisonRenderer(ImageRenderer):
    def render(self, original, reconstruction, vmin=0, vmax=255):
        fig, axs = plt.subplots(2, 1, figsize=(20, 11), gridspec_kw={'height_ratios': [2, 2]})

        axs[0].imshow(original, vmin=vmin, vmax=vmax, cmap='gray')
        axs[1].imshow(reconstruction, vmin=vmin, vmax=vmax, cmap='gray')

        fig.show()

        return fig


class DecisionBoundaryRenderer(ImageRenderer):
    def render(self, original, reconstruction, vmin=0, vmax=255):
        fig = plt.figure(figsize=(12, 5))

        plt.imshow(original, cmap='jet', vmin=vmin, vmax=vmax, interpolation='none')
        plt.imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')

        fig.show()

        return fig


class ComparisonWithOverlayRenderer(ImageRenderer):
    def render(self, original, reconstruction, vmin=0, vmax=1):
        fig, axs = plt.subplots(1, 3, figsize=(17, 9))

        axs[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)

        axs[1].imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax)

        axs[2].imshow(original, cmap='jet', vmin=vmin, vmax=vmax, interpolation='none')
        axs[2].imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')

        plt.show()


# reshape y_pred array from Keras into the right shape and dimensions for visualisation with matplotlib
def keras_to_matlibplot(y_pred):
    y_pred_copy = np.copy(y_pred)
    y_pred_flatten = []

    for x in range(y_pred.shape[0]):
        row = []

        for y in range(y_pred.shape[1]):
            row.append(y_pred_copy[x, y, 0, 0])

        y_pred_flatten.append(row)

    return np.asarray(y_pred_flatten)



