# centre-radius representation of AABB
# storage efficient, easy update, can be tested as a bounding sphere as well

from dataclasses import dataclass

from src.model.boundingvolume.BoundingVolume import BoundingVolume


@dataclass
class Point2D:
    """Represents a point in R2"""
    x: int
    y: int


@dataclass
class Point3D:
    """Represents a point in R3"""
    x: int
    y: int
    z: int


# cx: 31, cy: 30
# rx: 12, ry: 14

@dataclass
class AABB2D(BoundingVolume):
    """An integer implementation of axis-aligned bounding box"""
    centre: Point2D  # center point of AABB
    radius: tuple[int, int, int]  # radius or halfwidth extents (rx, ry, rz)

    # region R = { (x, y, z) | |c.x-x|<=rx, |c.y-y|<=ry, |c.z-z|<=rz }
    def intersection_test(self, point: Point2D) -> bool:
        intersected_x = abs(self.centre.x - point.x) <= self.radius[0]
        # print(abs(self.centre.x - point.x), self.radius[0])
        intersected_y = abs(self.centre.y - point.y) <= self.radius[1]

        return intersected_x and intersected_y


@dataclass
class AABB3D:
    """An integer implementation of axis-aligned bounding box"""
    centre: Point3D  # center point of AABB
    radius: tuple[int, int, int]  # radius or halfwidth extents (rx, ry, rz)

    # region R = { (x, y, z) | |c.x-x|<=rx, |c.y-y|<=ry, |c.z-z|<=rz }
    def intersection_test(self, point: Point3D) -> bool:
        intersected_x = abs(self.centre.x - point.x) <= self.radius[0]
        intersected_y = abs(self.centre.y - point.y) <= self.radius[1]
        intersected_z = abs(self.centre.z - point.z) <= self.radius[2]

        return intersected_x and intersected_y and intersected_z
