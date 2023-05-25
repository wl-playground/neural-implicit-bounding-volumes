import time

from src.model.kerasmlp.BinaryClassification import BinaryClassification


# instrument KerasMLP BinaryClassification model class with metrics
class BinaryClassificationWithMetrics(BinaryClassification):
    def __init__(self, optimiser, loss, hidden_layers, width, input_dimensions, output_dimensions, metrics_registry):
        super().__init__(optimiser, loss, hidden_layers, width, input_dimensions, output_dimensions, metrics_registry)

    def build(self):
        # register metric for build time
        start_time = time.perf_counter()
        self.metrics_registry.register_counter_metric("build")

        self.model.build()

        elapsed_time = time.perf_counter() - start_time
        self.metrics_registry.add("build", elapsed_time)

    def compile(self):
        # register metric for compile time

        start_time = time.perf_counter()
        self.metrics_registry.register_counter_metric("compile")

        self.model.compile()

        elapsed_time = time.perf_counter() - start_time
        self.metrics_registry.add("compile", elapsed_time)

    def train(self, x_train, y_target, batch_size, epochs, verbose_mode, class_weight=None):
        # register metric for train time

        start_time = time.perf_counter()
        self.metrics_registry.register_counter_metric("train")

        self.model.train(x_train, y_target, batch_size, epochs, verbose_mode, class_weight)

        elapsed_time = time.perf_counter() - start_time
        self.metrics_registry.add("train", elapsed_time)

    def validate(self):
        pass

    def test(self):
        pass

    def inference(self, input_value, verbose_mode=1):
        # register metric for inference time
        start_time = time.perf_counter()
        self.metrics_registry.register_counter_metric("inference")

        result = self.model.inference(input_value, verbose_mode)

        elapsed_time = time.perf_counter() - start_time
        self.metrics_registry.add("inference", elapsed_time)

        # register metrics for model binary classification performance
        self.metrics_registry.register_counter_metric("true_negative")
        self.metrics_registry.register_counter_metric("true_positive")
        self.metrics_registry.register_counter_metric("false_negative")
        self.metrics_registry.register_counter_metric("false_positive")

        for row in range(input_value.shape[0]):
            for col in range(input_value.shape[1]):
                if result[row, col] == 0 and input_value[row, col] == 0:
                    self.metrics_registry.increment_counter_metric("true_negative")
                elif result[row, col] == 1 and input_value[row, col] == 1:
                    self.metrics_registry.increment_counter_metric("true_positive")
                elif result[row, col] == 0 and input_value[row, col] == 1:
                    self.metrics_registry.increment_counter_metric("false_negative")
                elif result[row, col] == 1 and input_value[row, col] == 0:
                    self.metrics_registry.increment_counter_metric("false_positive")
                else:
                    raise ValueError("data error, unknown y_pred and y_target combination")

        return result

    def save_weights(self, filename='localWeights.h5'):
        self.model.save_weights(filename)

    def download_weights(self, filename='localWeights.h5'):
        self.model.download_weights(filename)