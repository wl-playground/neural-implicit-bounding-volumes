from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def train(
            self,
            x_train,
            y_target,
            class_weight,
            batch_size,
            epochs,
            verbose_mode
    ):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def inference(self, input_value, verbose_mode=1):
        pass

    @abstractmethod
    def save_weights(self, filename='localWeights.h5'):
        self.model.save_weights(filename)

    @abstractmethod
    def download_weights(self, filename='localWeights.h5'):
        files.download(filename)
