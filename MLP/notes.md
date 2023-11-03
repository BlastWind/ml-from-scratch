- Inputs are the first layer's activations, $a^{(1)}$.


- Example of computing the activation for a hidden layer's neuron (neuron 3 of layer 2).
$$
a_3^{(2)} = f( W_{31}^{(2)} a_1^{(1)} + W_{32}^{(2)} a_2^{(2)} + \cdots + b_3 ^{(1)}    )
$$
where $f$ is some activiation function, e.g., $tanh, relu$. Note additionally that $b_i^{(l)}$ is actually the bias associated with unit $i$ in layer $l + 1$. Hence, I think it's fine to store the biases in some `Layer` object, along with the weights.

- I'll 