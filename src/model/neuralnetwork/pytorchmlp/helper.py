import numpy as np
import torch
from prettytable import PrettyTable

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        table.add_row([name, params])
        total_params += params

    print(table)

    print(f"Total Trainable Params: {total_params}")

    return total_params


def reconstruct_image(input_value, model):
    y_pred = []

    for x in range(input_value.shape[0]):
        row = []

        for y in range(input_value.shape[1]):
            value = model.inference(torch.tensor([[float(x), float(y)]]).to(device))

            if value >= 0.5:
                row.append(1)
            else:
                row.append(0)

        y_pred.append(row)

    y_pred = np.asarray(y_pred)

    return y_pred