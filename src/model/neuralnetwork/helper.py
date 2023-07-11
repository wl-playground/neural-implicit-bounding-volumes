def hut_3d_calculator(image_width, image_height):
    if image_width is not image_height:
        raise ValueError("training image width and height must be the same for this calculator")

    scaling = 64.0/float(image_width)

    # max(0, -0.2*X - 0.2*Y + 13) calculated for 64x64 input
    f = "f = max(0, -{}*X - {}*Y + 13)".format(0.2*scaling, 0.2*scaling)

    # max(0, 0.2*X + 0.2*Y - 13) calculated for 64x64 input
    g = "g = max(0, {}*X + {}*Y - 13)".format(0.2*scaling, 0.2*scaling)

    # max(0, -0.2*X + 0.2*Y + 5) calculated for 64x64 input
    # make 5 bigger for a bigger hut, and smaller for a smaller hut
    h = "h = max(0, -{}*X + {}*Y + 5)".format(0.2*scaling, 0.2*scaling)

    # max(0, -0.4*X + 0.4*Y + 0) calculated for 64x64 input
    j = "j = max(0, -{}*X + {}*Y)".format(0.4*scaling, 0.4*scaling)

    # 3d hut
    k = "hut_3d = max(0, h - j - g - f)"

    print("for a training image of dimensions ({}, {}):\n".format(image_width, image_height))
    print(f)
    print(g)
    print(h)
    print(j, "\n")
    print(k)


# register metrics for evaluating model binary classification performance
def calculate_metrics(metrics_registry, input_value, result):
    metrics_registry.register_counter_metric("true_negative")
    metrics_registry.register_counter_metric("true_positive")
    metrics_registry.register_counter_metric("false_negative")
    metrics_registry.register_counter_metric("false_positive")

    for row in range(input_value.shape[0]):
        for col in range(input_value.shape[1]):
            if result[row, col] == 0 and input_value[row, col] == 0:
                metrics_registry.increment_counter_metric("true_negative")
            elif result[row, col] == 1 and input_value[row, col] == 1:
                metrics_registry.increment_counter_metric("true_positive")
            elif result[row, col] == 0 and input_value[row, col] == 1:
                metrics_registry.increment_counter_metric("false_negative")
            elif result[row, col] == 1 and input_value[row, col] == 0:
                metrics_registry.increment_counter_metric("false_positive")
            else:
                raise ValueError("data error, unknown y_pred and y_target combination")


def count_class_instances(y_target):
    positive_class = 0
    negative_class = 0

    for row in range(y_target.shape[0]):
        for col in range(y_target.shape[1]):
            if y_target[row, col] == 0.0:
                negative_class += 1

            if y_target[row, col] == 1.0:
                positive_class += 1

    return {
        "positive_class": positive_class,
        "negative_class": negative_class,
        "total_samples": y_target.shape[0] * y_target.shape[1]
    }
