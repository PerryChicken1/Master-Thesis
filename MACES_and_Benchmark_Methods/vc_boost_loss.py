from abc import ABC, abstractmethod
import numpy as np
from sklearn.utils.stats import _weighted_percentile

# File from https://github.com/sply88/vcboost
# loss function for compatibility with VC Boost implementation

class LossFunction(ABC):

    @abstractmethod
    def __call__(self, y, y_hat):
        raise NotImplemented

    @abstractmethod
    def negative_gradient(self, y, y_hat):
        raise NotImplemented

    @abstractmethod
    def line_search(self, residual, direction):
        raise NotImplemented


class LS(LossFunction):
    """Least squares loss"""

    def __call__(self, y, y_hat):
        residual = y - y_hat
        return np.dot(residual, residual) / residual.shape[0]

    def negative_gradient(self, y, y_hat):
        return y - y_hat

    def line_search(self, residual, direction):

        sqnorm = np.dot(direction, direction)

        if sqnorm < 1e-8:
            return 0
        else:
            return (1 / sqnorm) * np.dot(direction, residual)