from src.model.kerasmlp.BinaryClassification import BinaryClassification
from src.metrics.MetricsRegistry import MetricsRegistry

if __name__ == "__main__":
    model = BinaryClassification(
        "adam",
        "bce",
        2,
        5,
        2,
        1
    )

    metricsRegistry = MetricsRegistry()
