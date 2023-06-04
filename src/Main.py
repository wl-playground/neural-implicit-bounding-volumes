from src.metrics.MetricsRegistry import MetricsRegistry

if __name__ == "__main__":

    metricsRegistry = MetricsRegistry()
    metricsRegistry.register_counter_metric("test")
    metrics = metricsRegistry.get_metrics()
    print(metrics)
