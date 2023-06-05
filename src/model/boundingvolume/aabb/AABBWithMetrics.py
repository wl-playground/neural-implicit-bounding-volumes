from src.model.boundingvolume.aabb.AABB import AABB2D, Point2D


class AABB2DWithMetrics(AABB2D):
    def __init__(self, centre, radius, data, metrics_registry):
        super().__init__(centre, radius)
        self.data = data
        self.metrics_registry = metrics_registry

    def calculate_metrics(self):
        self.metrics_registry.register_counter_metric("true_negative")
        self.metrics_registry.register_counter_metric("true_positive")

        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                coordinate = Point2D(x=row, y=col)
                if self.intersection_test(coordinate):
                    coordinate_value = self.data[row, col]

                    if coordinate_value == 0:
                        self.metrics_registry.increment_counter_metric("true_negative")
                    elif coordinate_value == 1:
                        self.metrics_registry.increment_counter_metric("true_positive")
                    else:
                        raise ValueError("Unknown pixel value for data, make sure data is normalised")


