from src.data.DataLoader import DataLoader
from src.metrics.MetricsRegistry import MetricsRegistry
from src.model.boundingvolume.aabb.AABB import Point2D
from src.model.boundingvolume.aabb.AABBWithMetrics import AABB2DWithMetrics
from src.model.boundingvolume.sphere.SphereWithMetrics import Sphere2DWithMetrics

if __name__ == "__main__":
    bunny = DataLoader.load_binary_image("/Users/wenxinliu/Documents/thesis/thesis/data/2D/new_target_65x65.png")
    normalised_bunny = DataLoader.normalise_train(bunny)

    metrics_registry = MetricsRegistry()

    # cx: 31, cy: 30
    # rx: 12, ry: 14

    # TODO: double check the coordinates and implementation
    # {'true_negative': 426.0, 'true_positive': 415.0}
    aabb = AABB2DWithMetrics(
        centre=Point2D(
            x=32,
            y=31
        ),
        radius=(15, 15),
        data=normalised_bunny,
        metrics_registry=metrics_registry
    )

    aabb.calculate_metrics()
    print(metrics_registry.get_metrics())

    metrics_registry = MetricsRegistry()

    # # TODO: double check the coordinates
    # # {'true_negative': 426.0, 'true_positive': 415.0}
    # sphere = Sphere2DWithMetrics(
    #     centre=Point2D(
    #         x=31,
    #         y=30
    #     ),
    #     radius=14,
    #     data=normalised_bunny,
    #     metrics_registry=metrics_registry
    # )
    #
    # sphere.calculate_metrics()
    # print(metrics_registry.get_metrics())


