import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # define initial empty neural network container
        self.criterion = None
        self.optimiser = None
        self.dataloader = None
        self.model = nn.Sequential()

    # setter method for incrementally building a neural network model
    # since we use a setter method, we can decouple this neural network class
    # declaration from any specific neural network architecture that an instance
    # of this class represents
    def add_layer(self, layer):
        self.model.append(layer)

    # getter method for model parameters
    # useful for when instantiating the optimiser
    def get_model_parameters(self):
        return self.parameters()

    # getter method for model summary
    def get_model_summary(self):
        return self

    # setter method for setting layer weights
    def set_weight(self, layer_index, weight):
        self.model[layer_index].weight = nn.parameter.Parameter(weight)

    # setter method for setting layer bias
    def set_bias(self, layer_index, bias):
        self.model[layer_index].bias = nn.parameter.Parameter(bias)

    # setter method for setting the dataloader instance
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    # setter method for setting the optimiser instance
    def set_optimiser(self, optimiser):
        self.optimiser = optimiser

    # setter method for setting the loss function instance
    def set_criterion(self, loss_function):
        self.criterion = loss_function

    def forward(self, x):
        x = self.model(x)

        return x

    def train(self, mode=True):
        # training for one epoch
        # call train() from a training loop to train for multiple epochs

        self.model.train(True)  # turn on autograd for training

        running_loss = 0
        last_loss = 0

        for index, (inputs, labels) in enumerate(self.dataloader):
            # zero out the gradient buffer before next batch
            self.optimiser.zero_grad()

            # get model prediction
            outputs = self.model(inputs.to(device))

            # compute loss and gradients
            loss = self.criterion(outputs, labels.to(device))
            loss.backward()

            # do gradient descent step
            self.optimiser.step()

            # get metrics
            running_loss += loss.item()

            if index % len(self.dataloader) == len(self.dataloader) - 1:
                last_loss = running_loss / len(self.dataloader)  # loss per batch

                running_loss = 0

        return last_loss

    def inference(self, x):
        if not torch.is_tensor(x):
            raise TypeError("input for inference needs to be torch.Tensor")

        self.model.eval()  # turn off autograd for inference

        result = self.model(x)

        return result.item()
