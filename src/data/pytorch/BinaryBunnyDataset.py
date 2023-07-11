import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryBunnyDataset(Dataset):
    def __init__(self):
        binary_bunny = np.loadtxt(
            fname='/content/neural-implicit-bounding-volumes/data/2D/binary_bunny_dataset.csv',
            delimiter=",",
            dtype=np.float32,
            skiprows=1
        )

        self.x_train = torch.from_numpy(binary_bunny[:, 0:2])
        self.y_target = torch.from_numpy(binary_bunny[:, [2]])
        self.number_of_samples = self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_target[index]

    def __len__(self):
        return self.number_of_samples
