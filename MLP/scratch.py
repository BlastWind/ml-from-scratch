from typing import Callable, List, Tuple
import numpy as np
from numpy.typing import NDArray
from functools import reduce

# ---------- BEGIN: Activation Functions ---------- #
def sigmoid(xs: NDArray):
	'''Squashes output between [0, 1]'''
	
	return 1 / 1 + np.exp(-xs)

def tanh(xs: NDArray):
    return np.tanh(xs)

def softmax(xs: NDArray):
    exp_x = np.exp(xs - np.max(xs))  # Subtracting np.max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def relu(xs: NDArray):
    return np.maximum(0, xs)

def leaky_relu(xs: NDArray, alpha: float = 0.01):
    return np.maximum(alpha * xs, xs)

def identity(a): return a
# ---------- END: Activation Functions ---------- #

# ---------- BEGIN: Initialization Functions for weights and biases ---------- #
def uniform_sample(layer_shape):
	in_, out = layer_shape
	return np.random.uniform(-1, 1, (out, in_))

def he_initialization(layer_shape):
    in_, out = layer_shape
    stddev = np.sqrt(2. / in_)
    return np.random.normal(0, stddev, (out, in_))

# ---------- END: Initialization Functions for weights and biases ---------- #

# ---------- BEGIN: Cost Functions ---------- #
def mse(y_pred: NDArray, y_true: NDArray):
	# Mean Squared Error
	return 0.5 * np.mean(np.square(y_true - y_pred))

# ---------- END: Cost Functions ---------- #
class Layer(): 
	def __init__(self, input_size: int, output_size: int, activation_function: Callable[[NDArray], NDArray], weight_initialization_function: Callable[[Tuple[int, int]], NDArray] = uniform_sample, bias_initialization_function: Callable[[int], NDArray] = np.zeros) -> None:
		self.input_size = input_size 
		self.output_size = output_size
		self.activation_function = activation_function
		self.weights = weight_initialization_function((input_size, output_size))
		self.bias = bias_initialization_function(input_size)


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
	
	def forward(self, inputs: NDArray):
		def compute(xs: NDArray, layer: Layer):
			# print(layer.weights.shape, xs.shape)
			return np.matmul(layer.weights, xs)
		return reduce(compute, self.layers, inputs)

	def train(self, data: List[Tuple[NDArray, NDArray]], cost_function: Callable[[NDArray, NDArray], float], learning_rate: float, regularization_parameter: float, save_path: str):
		'''Trains a network using the specified `data`, then outputs it somewhere...

		Args:
			data: A list of pairs of training input and output.
			cost_function: Takes in the network output and ground truth to produce the scalar cost.
			regularization_parameter: Note, networks always train using L2 Regularization, whose only parameter you provide 
			(usually on a logarithmic scale, i.e., 0.01, 0.1, 1, 10, ...), value depends on k-fold cross-validation results.
		'''


layers = [Layer(4, 20, leaky_relu, uniform_sample), Layer(20, 10, leaky_relu, uniform_sample), Layer(10, 2, softmax, uniform_sample)]
n = Network(layers)
n.validate()
res = n.forward(np.array([1, 2, 3, 4]))
print(res.shape)