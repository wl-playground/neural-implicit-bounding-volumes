from enum import Enum

from src.view.ImageRenderer import ComparisonRenderer, DecisionBoundaryRenderer


class ImageVisualisationType(Enum):
    COMPARISON = 0
    DECISIONBOUNDARY = 1


factory_instances = {}


def get_renderer(visualisation_type):
    if visualisation_type in factory_instances.keys():
        return factory_instances[visualisation_type]
    else:
        renderer = _get_renderer(visualisation_type)
        factory_instances[visualisation_type] = renderer

    return renderer


def _get_renderer(visualisation_type):
    if visualisation_type == ImageVisualisationType.COMPARISON:
        return ComparisonRenderer()
    elif visualisation_type == ImageVisualisationType.DECISIONBOUNDARY:
        return DecisionBoundaryRenderer()
    else:
        raise ValueError("visualisation type {} not yet supported".format(visualisation_type))
