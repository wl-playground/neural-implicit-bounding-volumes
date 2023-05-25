from enum import Enum

from src.view.ImageRenderer import ComparisonRenderer, DecisionBoundaryRenderer


class ImageVisualisationType(Enum):
    COMPARISON = 0
    DECISIONBOUNDARY = 1


class RendererFactory:
    def __init__(self):
        self.factory_instances = {}

    def get(self, visualisation_type):
        if visualisation_type in self.factory_instances.keys():
            return self.factory_instances[visualisation_type]
        else:
            renderer = self._get_renderer(visualisation_type)
            self.factory_instances[visualisation_type] = renderer

        return renderer

    def _get_renderer(self, visualisation_type):
        if visualisation_type == ImageVisualisationType.COMPARISON:
            return ComparisonRenderer()
        elif visualisation_type == ImageVisualisationType.DECISIONBOUNDARY:
            return DecisionBoundaryRenderer()
        else:
            raise ValueError("visualisation type {} not yet supported".format(visualisation_type))
