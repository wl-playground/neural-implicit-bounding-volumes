from enum import Enum
from abc import ABC, abstractmethod


class Renderer(ABC):
    @abstractmethod
    def render(self, original, reconstruction):
        pass


class ComparisonRenderer(Renderer):
    def render(self, original, reconstruction):
        pass


class DecisionBoundaryRenderer(Renderer):
    def render(self, original, reconstruction):
        pass


class VisualisationType(Enum):
    COMPARISON = 0
    DECISIONBOUNDARY = 1


class RendererFactory:
    def __init__(self):
        self.factory_instances = {}

    @staticmethod
    def get(self, visualisation_type):
        if visualisation_type in self.factory_instances.keys():
            return self.factory_instances[visualisation_type]
        else:
            renderer = self._get_renderer(visualisation_type)
            self.factory_instances[visualisation_type] = renderer

        return renderer

    def _get_renderer(self, visualisation_type):
        if visualisation_type == VisualisationType.COMPARISON:
            return ComparisonRenderer()
        elif visualisation_type == VisualisationType.DECISIONBOUNDARY:
            return DecisionBoundaryRenderer()
        else:
            raise ValueError("visualisation type {} not yet supported".format(visualisation_type))
