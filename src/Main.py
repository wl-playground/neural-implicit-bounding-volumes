from src.data.ImageDataLoader import ImageDataLoader
from src.metrics.MetricsRegistry import MetricsRegistry
from src.model.boundingvolume.aabb.AABB import Point2D, AABBMinWidth2D, AABBCentreRadius2D
from src.model.scene.Scene2D import Scene2D
from src.view import RendererFactory
from src.view.RendererFactory import ImageVisualisationType

if __name__ == "__main__":
    bunny = ImageDataLoader.load_binary_image("/Users/wenxinliu/Documents/thesis/neural-implicit-bounding-volumes/data/2D/new_target_64x64.png")
    normalised_bunny = ImageDataLoader.normalise_train(bunny)

    metrics_registry = MetricsRegistry()

    # aabb = AABBCentreRadius2D(
    #     centre=Point2D(x=32, y=32),
    #     radius=(15, 13)
    # )

    # TODO: orientation of x and y not correct in the implementation
    # {'true_negative': 365.0, 'true_positive': 415.0}
    aabb = AABBMinWidth2D(
        min=Point2D(x=17, y=20),
        diameter=(29, 25)
    )

    scene = Scene2D(
        bounding_volume=aabb,
        rendering_geometry=normalised_bunny,
        metrics_registry=metrics_registry
    )

    data = scene.calculate_scene_data_min_widths()
    scene.calculate_scene_metrics()

    renderer = RendererFactory.get_renderer(ImageVisualisationType.SCENE2D)
    renderer.render(normalised_bunny, data, 0, 1)

    print(metrics_registry.get_metrics())
