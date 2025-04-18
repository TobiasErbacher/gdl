from abc import ABC, abstractmethod

import numpy as np


class TrainArgs(ABC):
    """
    Contains properties needed in the train_epoch() method.
    Every subclass must have the learning_rate property.
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class EvalArgs(ABC):
    """
    Contains properties needed in the evaluate() method.
    """
    pass


class ModelArgs(ABC):
    """
    The properties of this class are passed to the constructor of the model.
    Thus, only include properties that are required by the model's constructor.
    """
    pass


class Integrator(ABC):
    @abstractmethod
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int, train_args: TrainArgs) -> (float, np.array):
        """
        Trains the model for one epoch on the data. The returned loss is only used for logging.
        :return: The loss and an array containing the steps per node of the training set
        """
        pass

    @abstractmethod
    def evaluate(self, model, data, mask, eval_args: EvalArgs = None) -> (float, np.array):
        """
        Evaluates the model on the provided data.
        :return: The accuracy on the test set and the steps per node of the test set
        """
        pass
