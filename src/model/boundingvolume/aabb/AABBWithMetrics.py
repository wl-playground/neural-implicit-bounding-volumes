from src.model.boundingvolume.BoundingVolumeWithData import BoundingVolumeWithData
from src.model.boundingvolume.BoundingVolumeWithMetrics import BoundingVolumeWithMetrics
from src.model.boundingvolume.aabb.AABB import AABB2D, Point2D


class AABB2DWithMetrics(AABB2D, BoundingVolumeWithData, BoundingVolumeWithMetrics):
    def __init__(self, centre, radius, data, metrics_registry):
        super().__init__(centre, radius)
        super().__init__(data)
        super().__init__(metrics_registry)


