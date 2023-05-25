class MetricsRegistry:
    def __int__(self):
        self.metrics_registry = {}

    def register_metric(self, name, default_value):
        self.metrics_registry[name] = default_value

    def record_metrics(self, y_pred, y_target):
        self.register_metric("true_negative", 0)
        self.register_metric("true_positive", 0)
        self.register_metric("false_negative", 0)
        self.register_metric("false_positive", 0)

        for row in range(y_pred.shape[0]):
            for col in range(y_pred.shape[1]):
                if y_pred[row, col] == 0 and y_target[row, col] == 0:
                    self.metrics_registry["true_negative"] += 1
                elif y_pred[row, col] == 1 and y_target[row, col] == 1:
                    self.metrics_registry["true_positive"] += 1
                elif y_pred[row, col] == 0 and y_target[row, col] == 1:
                    self.metrics_registry["false_negative"] += 1
                elif y_pred[row, col] == 1 and y_target[row, col] == 0:
                    self.metrics_registry["false_positive"] += 1
                else:
                    raise Exception("data error, unknown y_pred and y_target combination")

    def get_metrics(self):
        return self.metrics_registry
