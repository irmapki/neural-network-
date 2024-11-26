# prompt: buatkan lah codingan neural network yang ada feed forward dan back propagation

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backpropagation(self, X, y, learning_rate):
        m = X.shape[0]  # Number of samples

        # Output layer error
        delta2 = (self.a2 - y) * self.sigmoid_derivative(self.a2)

        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 -= learning_rate * np.dot(self.a1.T, delta2) / m
        self.b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True) / m
        self.W1 -= learning_rate * np.dot(X.T, delta1) / m
        self.b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True) / m

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

input_size = 2
hidden_size = 2
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
  nn.feedforward(X)
  nn.backpropagation(X, y, learning_rate)

print("Final weights and biases:")
print("W1:", nn.W1)
print("b1:", nn.b1)
print("W2:", nn.W2)
print("b2:", nn.b2)
print("\nPredictions:")
predictions = nn.feedforward(X)
predictions