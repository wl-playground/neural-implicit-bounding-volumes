import copy
import numpy as np
from PIL import Image


class DataLoader:
    @staticmethod
    def load_binary_image(filename):
        image = Image.open(filename).convert('L')

        # convert grayscale image into a numpy array
        return np.array(image)

    # normalise input data from 0, 255 to 0, 1 to prepare for binary classification
    @staticmethod
    def normalise_train(array_2d):
        result = copy.deepcopy(array_2d)

        for row in range(result.shape[0]):
            for col in range(result.shape[1]):
                if result[row, col] == 255:
                    result[row, col] = 1

        return result

    # reshape x of the training dataset
    @staticmethod
    def get_x_train(image_width, image_height):
        x_train = []

        for x in range(image_width):
            for y in range(image_height):
                coord = [x, y]

                x_train.append(coord)

        return np.asarray(x_train)

    # reshape y_target of the training dataset from 2D array to 1D array
    @staticmethod
    def get_y_target(array_2d):
        y_target = []

        for x in range(array_2d.shape[0]):
            for y in range(array_2d.shape[1]):
                pixel_value = array_2d[x, y]

                y_target.append(pixel_value)

        return np.asarray(y_target)
