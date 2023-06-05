from abc import ABC
import numpy as np

from src.model.boundingvolume.BoundingVolume import BoundingVolume
from src.model.boundingvolume.aabb.AABB import Point2D


class BoundingVolumeWithData(ABC, BoundingVolume):
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.calculate_data()

    def calculate_data(self):
        y_pred = []

        for row in range(self.data.shape[0]):
            row = []

            for col in range(self.data.shape[1]):
                coordinate = Point2D(x=row, y=col)

                if self.intersection_test(coordinate):
                    coordinate_value = self.data[row, col]

                    row.append(coordinate_value)
            y_pred.append(row)

        return np.asarray(y_pred)




