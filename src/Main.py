from src.KerasBinaryClassificationMLP import KerasBinaryClassificationMLP
from src.MetricsRegistry import MetricsRegistry

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
