from src.model.kerasmlp.KerasBinaryClassificationMLP import KerasBinaryClassificationMLP
from src.metrics.MetricsRegistry import MetricsRegistry

if __name__ == "__main__":
    model = KerasBinaryClassificationMLP(
        "adam",
        "bce",
        2,
        5,
        2,
        1
    )

    metricsRegistry = MetricsRegistry()
