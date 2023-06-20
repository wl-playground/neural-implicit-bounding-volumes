from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt

import numpy as np


class ImageRenderer(ABC):
    @abstractmethod
    def render(self, original, reconstruction, vmin, vmax):
        pass

    def model_output_to_matlibplot(self, y_pred):
        # reshape y_pred array into the right shape and dimensions for
        # visualisation with matplotlib
        y_pred_copy = np.copy(y_pred)
        y_pred_flatten = []

        for x in range(y_pred.shape[0]):
            row = []

            for y in range(y_pred.shape[1]):
                row.append(y_pred_copy[x, y, 0, 0])

            y_pred_flatten.append(row)

        return np.asarray(y_pred_flatten)


class ComparisonRenderer(ImageRenderer):
    def render(self, original, reconstruction, vmin=0, vmax=255):
        matplotlib.rc('image', cmap='gray')

        reconstruction = super().model_output_to_matlibplot(reconstruction)

        fig, axs = plt.subplots(2, 1, figsize=(20, 11), gridspec_kw={'height_ratios': [2, 2]})

        axs[0].imshow(original, vmin=vmin, vmax=vmax)
        axs[1].imshow(reconstruction, vmin=vmin, vmax=vmax)

        plt.show()


class DecisionBoundaryRenderer(ImageRenderer):
    def render(self, original, reconstruction, vmin=0, vmax=255):
        reconstruction = super().model_output_to_matlibplot(reconstruction)

        plt.figure(figsize=(12, 5))

        plt.imshow(original, cmap='jet', vmin=vmin, vmax=vmax, interpolation='none')

        plt.imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')

        plt.show()


class SceneRenderer2D(ImageRenderer):
    def render(self, original, reconstruction, vmin, vmax):
        # plt.imshow(original, cmap='jet', vmin=vmin, vmax=vmax, interpolation='none')

        # plt.imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none', extent=[40, 60, 60, 80])

        plt.imshow(reconstruction, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')

        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        #
        # fig, ax = plt.subplots()
        #
        # imagebox = OffsetImage(reconstruction, zoom=0.4)
        # imagebox.image.axes = ax
        #
        # ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='axes fraction',
        #                     bboxprops={'lw':0})
        #
        # ax.add_artist(ab)

        plt.show()

