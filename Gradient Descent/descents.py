from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """

        if self.loss_function is LossFunction.MSE:
            loss = (1 / x.shape[0]) * (y - x @ self.w).T @ (y - x @ self.w)
        elif self.loss_function is LossFunction.LogCosh:
            loss = (np.mean(np.log(np.cosh(y - x @ self.w))))
        elif self.loss_function is LossFunction.MAE:
            loss = np.mean(np.abs(y - x @ self.w))
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0 
            loss = np.mean(np.where(np.abs(y - x @ self.w) <= delta, 0.5 * (y - x @ self.w) ** 2, delta * (np.abs(y - x @ self.w) - 0.5 * delta)))
        else:
            raise NotImplementedError('BaseDescent calc_loss function not implemented')

        return loss



    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        y_pred = np.dot(x, self.w)
        return y_pred

        # TODO: implement prediction function
        # raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        eta = self.lr()
        weight_difference = -eta * gradient
        self.w = self.w + weight_difference
        return weight_difference
        # TODO: implement updating weights function

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(x)  # Это должно быть вычислено перед использованием в любом блоке

        if self.loss_function is LossFunction.MSE:
            gradient = (-2/x.shape[0]) * np.dot(x.T, (y - y_pred))
        elif self.loss_function is LossFunction.LogCosh:
            gradient = (-1/x.shape[0]) * np.dot(np.tanh(y - y_pred), x)
        elif self.loss_function is LossFunction.MAE:
            gradient = (-1/x.shape[0]) * np.dot(np.sign(y - y_pred), x)
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0  # Huber loss threshold
                # Для Huber loss, градиент отличается для ошибок меньше delta и больше delta
            huber_grad = np.where(np.abs(y - y_pred) <= delta, 
                                      y - y_pred,
                                      delta * np.sign(y - y_pred))
            gradient = (-1/x.shape[0]) * np.dot(huber_grad, x)
        else:
            raise NotImplementedError(f'Gradient for loss function {self.loss_function} not implemented')

        return gradient

class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # TODO: implement calculating gradient function
        batch_indices = np.random.randint(0, x.shape[0], self.batch_size)
        return super().calc_gradient(x[batch_indices, :], y[batch_indices])
        # raise NotImplementedError('StochasticDescent calc_gradient function not implemented')


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        
        # TODO: implement updating weights function
        eta = self.lr()
        self.h = self.alpha * self.h + eta * gradient
        weight_difference = -self.h
        self.w = self.w + weight_difference
        
        return weight_difference
        # raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        # TODO: implement updating weights function
        self.iteration += 1
        eta = self.lr()

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient**2)

        m_hat = self.m / (1 - self.beta_1**self.iteration)
        v_hat = self.v / (1 - self.beta_2**self.iteration)

        weight_difference = - (eta / (np.sqrt(v_hat) + self.eps)) * m_hat
        self.w = self.w + weight_difference

        return weight_difference
        # raise NotImplementedError('Adagrad update_weights function not implemented')

class Nadam(VanillaGradientDescent):
    """
    Nadam Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        eta = self.lr()

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)

        weight_difference = -eta * (1 / np.sqrt(v_hat + self.eps)) * (self.beta_1 * m_hat + ((1 - self.beta_1) / (1 - self.beta_1 ** self.iteration)) * gradient)
        self.w = self.w + weight_difference

        return weight_difference
        # raise NotImplementedError('Adagrad update_weights function not implemented')

class AdaMax(VanillaGradientDescent):
    """
    AdaMax gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8  # Epsilon value for numerical stability
        self.m: np.ndarray = np.zeros(dimension)  # First moment vector
        self.u: np.ndarray = np.zeros(dimension)  # Scaled infinity norm (replaces v in Adam)
        self.beta_1: float = 0.9  # Exponential decay rate for first moment
        self.beta_2: float = 0.999  # Exponential decay rate for second moment (u)
        self.iteration: int = 0  # Time step

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights using the AdaMax optimization algorithm.
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        eta = self.lr() 

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        self.u = np.maximum(self.beta_2 * self.u, np.abs(gradient))

        m_hat = self.m / (1 - self.beta_1 ** self.iteration)

        weight_difference = - (eta / (self.u + self.eps)) * m_hat
        self.w = self.w + weight_difference

        return weight_difference
    
class AMSGrad(VanillaGradientDescent):
    """
    AMSGrad gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8
        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_hat: np.ndarray = np.zeros(dimension)  # The maximum of the v's up to the current time step
        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.iteration += 1
        eta = self.lr()

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        self.v_hat = np.maximum(self.v_hat, self.v)  # Update v_hat to be the max of previous v's

        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        weight_difference = - (eta / (np.sqrt(self.v_hat) + self.eps)) * m_hat
        self.w = self.w + weight_difference

        return weight_difference

    
class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient = self.w
        
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """
                                                                
class NadamReg(BaseDescentReg, Nadam):
    """
    Adaptive gradient algorithm with regularization class
    """    
class AdaMaxReg(BaseDescentReg, AdaMax):
    """
    Adaptive gradient algorithm with regularization class
    """  
class AMSGradReg(BaseDescentReg, AMSGrad):
    """
    Adaptive gradient algorithm with regularization class
    """  


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': Nadam if not regularized else NadamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg,
        'amsgrad': AMSGrad if not regularized else AMSGradReg}

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
