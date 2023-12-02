from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from functools import reduce

# ---------- BEGIN: Activation Functions ---------- #
def sigmoid(xs: NDArray[np.floating[Any]]):
	'''Squashes output between [0, 1]'''
	
	return 1 / 1 + np.exp(-xs)

def tanh(xs: NDArray[np.floating[Any]]):
    return np.tanh(xs)

def softmax(xs: NDArray[np.floating[Any]]):
    exp_x = np.exp(xs - np.max(xs))  # Subtracting np.max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def relu(xs: NDArray[np.floating[Any]]):
    return np.maximum(0, xs)

def leaky_relu(xs: NDArray[np.floating[Any]], alpha: float = 0.01):
    return np.maximum(alpha * xs, xs)

def leaky_relu_derivative(xs: NDArray[np.floating[Any]], alpha=0.01):
	return np.where(xs > 0, 1, alpha)

def identity(a: Any): return a
# ---------- END: Activation Functions ---------- #

# ---------- BEGIN: Initialization Functions for weights and biases ---------- #
def uniform_sample(layer_shape) ->  NDArray[np.floating[Any]]:
	in_, out = layer_shape
	return np.random.uniform(-1, 1, (out, in_))

def he_initialization(layer_shape)  -> NDArray[np.floating[Any]]:
    in_, out = layer_shape
    stddev = np.sqrt(2. / in_)
    return np.random.normal(0, stddev, (out, in_))

# ---------- END: Initialization Functions for weights and biases ---------- #

# ---------- BEGIN: Cost Functions ---------- #
# def mse(y_pred: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]) -> np.floating[Any]:
# 	# Mean Squared Error
# 	return 0.5 * np.mean(np.square(y_true - y_pred))

def mse_derivative(y_pred: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
	return y_pred - y_true

# ---------- END: Cost Functions ---------- #
class Layer(): 
	def __init__(self, input_size: int, output_size: int, activation_function: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]], activation_function_derivative: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]], weight_initialization_function: Callable[[Tuple[int, int]], NDArray[np.floating[Any]]] = uniform_sample, bias_initialization_function: Callable[[int], NDArray[np.floating[Any]]] = np.zeros) -> None:
		self.input_size = input_size 
		self.output_size = output_size
		self.activation_function = activation_function
		self.weights = weight_initialization_function((input_size, output_size))
		self.bias = bias_initialization_function(output_size)
		self.last_weighted_sum = np.zeros(output_size)
		self.last_activation = np.zeros(output_size)
		self.activation_function_derivative = activation_function_derivative


	def forward(self, input: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
		weighted_sum = np.matmul(self.weights, input) + self.bias
		activation = self.activation_function(weighted_sum)
		self.last_weighted_sum = weighted_sum
		self.last_activation = activation
		return activation

	def backprop(self, prevLayer: Optional[Layer], nextLayer: Optional[Layer], y_pred: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]], input: NDArray[np.floating[Any]], cost_function_derivative: Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],NDArray[np.floating[Any]] ],):
		# The direction of the layers is still viewed from left to right. So, the next layer is always closer to the outer layer

		dzb = np.ones(self.output_size) # No use since just 1's.

		if not nextLayer: # Is the last layer
			assert prevLayer

			dzw = np.tile(prevLayer.last_activation, (prevLayer.last_activation.shape[0], 1))		

			daz = self.activation_function_derivative(self.last_weighted_sum)
			self.daz = daz
			dca = cost_function_derivative(y_pred, y_true)
			self.dca = dca
			dcw = np.matmul( np.transpose(dca * daz), dzw)
			dcb = dca * daz

			return (dcw, dcb)
			
		else: # If initial or hidden Layer
			
			dzw: NDArray[np.floating[Any]] = np.tile(prevLayer.last_activation, (prevLayer.last_activation.shape[0], 1)) if prevLayer else input	
			daz = self.activation_function_derivative(self.last_weighted_sum)
			self.daz = daz
			next_dca = nextLayer.dca 
			next_daz = nextLayer.daz 
			dca = np.sum(np.transpose(next_dca * next_daz), axis=0)
			self.dca = dca 
			dcw = np.matmul( np.transpose(dca * daz), dzw)
			dcb = dca * daz

			return (dcw, dcb)

class Network():
	def __init__(self, layers: List[Layer]):
		self.layers = layers
		if not self.validate():
			print("Network will not work, layer dimensions are wrong.")
			exit()

	def validate(self):
		if len(self.layers) == 0: return False
		for l1, l2 in zip(self.layers, self.layers[1:]):
			if l1.output_size != l2.input_size: return False
		return True
	
	def forward(self, inputs: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
		return reduce(lambda xs, l: l.forward(xs), self.layers, inputs)

	def train(self, data: List[Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]], learning_rate: float, regularization_parameter: float,  epochs: int, save_path: str):
		'''Trains a network using the specified `data`, then outputs it somewhere...

		Args:
			data: A list of pairs of training input and output.
			cost_function: Takes in the network output and ground truth to produce the scalar cost. Operates on just one result.
			regularization_parameter: Note, networks always train using L2 Regularization, whose only parameter you provide 
			(usually on a logarithmic scale, i.e., 0.01, 0.1, 1, 10, ...), value depends on k-fold cross-validation results.
		'''
		if len(data) == 0: return 
		if len(self.layers) <2: 
			print("Currently only work with networks with >=2 layers")
			return

		for _ in range(epochs):
			# total_cost = 0
			
			
			# forward, get cost
			for (x, y_true) in data:
				y_pred = self.forward(x)
				
				for i, layer in reversed(list(enumerate(self.layers))):
					print(f"Invoked backprop for layer {i}")
					(dcw, dcb) = layer.backprop(self.layers[i-1] if i != 0 else None, self.layers[i+1] if i != len(self.layers) - 1 else None, y_pred, y_true, x, mse_derivative)
					print(f"Finished backprop for layer {i}")
					layer.acc_dcw = layer.acc_dcw if not layer.acc_dcw else layer.acc_dcw + dcw 
					layer.acc_dcb = layer.acc_dcb if not layer.acc_dcb else layer.acc_dcb + dcb 


				# for layer in self.layers[1::-1]: # From second to last, to first.
			for layer in self.layers:
				final_dcw, final_dcb = layer.acc_dcw / len(data), layer.acc_dcb / len(data)
				layer.acc_dcw = layer.acc_dcb = None
				layer.weights -= final_dcw
				layer.bias -= final_dcb
			

			# backward, update weights
			# total_cost = (total_cost / len(data)) + regularization_parameter / 2 * sum(np.sum(np.square(layer.weights)) for layer in self.layers) 


layers = [Layer(4, 20, leaky_relu, leaky_relu_derivative, uniform_sample), Layer(20, 10, leaky_relu, leaky_relu_derivative, uniform_sample), Layer(10, 2, softmax, leaky_relu_derivative, uniform_sample)]
n = Network(layers)
n.validate()
# print(res.shape)

n.train([(np.random.random(4), np.random.random(2)), (np.random.random(4), np.random.random(2)), (np.random.random(4), np.random.random(2))], learning_rate=0.05, regularization_parameter=0, epochs=1, save_path="")