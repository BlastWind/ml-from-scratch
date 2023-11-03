from typing import List
import numpy as np
from numpy.typing import NDArray
from functools import reduce

class Layer(): 
	def __init__(self, input_size: int, output_size: int) -> None:
		self.input_size = input_size 
		self.output_size = output_size
		self.weights = np.zeros((output_size, input_size))
		self.bias = np.zeros(input_size)

class Network():
	def __init__(self, layers: List[Layer]):
		self.layers = layers
		if not self.validate():
			print("Network will not work, layer dimensions are wrong.")

	def validate(self):
		if len(self.layers) == 0: return False
		for l1, l2 in zip(self.layers, self.layers[1:]):
			if l1.output_size != l2.input_size: return False
		return True
	
	def forward(self, inputs: NDArray[np.float64]):
		def compute(xs: NDArray[np.float64], layer: Layer):
			# print(layer.weights.shape, xs.shape)
			return np.matmul(layer.weights, xs)

		return reduce(compute, self.layers, inputs)

layers = [Layer(4, 5), Layer(5, 2)]
n = Network(layers)
n.validate()
res = n.forward(np.array([1, 2, 3, 4]))
print(res.shape)