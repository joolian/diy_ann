"""
Simple artificial neural network. Code was adapted from:
https://github.com/harshitkgupta/StudyMaterial/blob/master/Make%20Your%20Own%20Neural%20Network%20(Tariq%20Rashid)%20-%20%7BCHB%20Books%7D.pdf
"""
import numpy as np


class HandMadeNeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, epochs, input_weights=None,
                 hidden_weights=None):
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._weights_input_hidden = input_weights
        self._weights_hidden_output = hidden_weights
        self._set_weights()

    def _set_weights(self):
        """Set the weights for the input and hidden layers"""
        if not all([
            isinstance(self._weights_input_hidden, np.ndarray),
            isinstance(self._weights_hidden_output, np.ndarray)
        ]):
            self._weights_input_hidden = np.random.rand(self._hidden_nodes, self._input_nodes) - 0.5
            self._weights_hidden_output = np.random.rand(self._output_nodes, self._hidden_nodes) - 0.5

    def _sigmoid_activation(self, values):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(- values))

    def _train(self, train, targets):
        """Train the neural network using a single instance"""
        inputs = train.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
        # Forward pass
        hidden_inputs = np.dot(self._weights_input_hidden, inputs)
        hidden_outputs = self._sigmoid_activation(hidden_inputs)
        output_inputs = np.dot(self._weights_hidden_output, hidden_outputs)
        final_outputs = self._sigmoid_activation(output_inputs)
        # Back propagation
        final_output_errors = targets - final_outputs
        hidden_errors = np.dot(self._weights_hidden_output.T, final_output_errors)
        # Gradient descent
        self._weights_hidden_output += self._learning_rate * np.dot(
            (final_output_errors * final_outputs * (1 - final_outputs)),
            np.transpose(hidden_outputs)
        )
        self._weights_input_hidden += self._learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            np.transpose(inputs)
        )

    def train(self, X_train, y_train):
        """
        Train the neural network
        :param X_train: The training values as a numpy array
        :param y_train: The labels for the training values as a numpy array
        :return:
        """
        for _ in range(self._epochs):
            for x, y in zip(X_train, y_train):
                targets = np.zeros(self._output_nodes) + 0.01
                targets[y] = 0.99
                self._train(x, targets)

    def _predict(self, input):
        """
        Predict using a single instance

        :param: the value of the inputs as a numpy array

        """
        hidden_inputs = np.dot(self._weights_input_hidden, input)
        hidden_outputs = self._sigmoid_activation(hidden_inputs)
        output_inputs = np.dot(self._weights_hidden_output, hidden_outputs)
        outputs = self._sigmoid_activation(output_inputs)
        return outputs

    def predict(self, inputs):
        """
        Predict multiple instances
        :param inputs: the values to predict
        :return: predicted values as a numpy array
        """
        predictions = []
        for x in inputs:
            predictions.append(np.argmax(self._predict(x)))
        return np.array(predictions)

    @property
    def hidden_weights(self):
        """ The weights for the connections between the hidden nodes and the output nodes"""
        return self._weights_hidden_output

    @property
    def input_weights(self):
        """The weights for the connections between the input nodes and the hidden nodes"""
        return self._weights_input_hidden
