from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BoundingVolume(ABC):
    @abstractmethod
    def intersection_test(self, coordinate):
        pass
