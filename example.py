from sklearn.datasets import fetch_openml
from diy_ann import HandMadeNeuralNetwork
import numpy as np


def scale_image(images):
    return images / 255 * 0.99 + 0.01


def accuracy(predicted, targets):
    scores = predicted == targets
    return np.count_nonzero(scores) / np.shape(targets)[0]


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    y = y.astype(int)
    X_scaled = scale_image(X)
    X_train, X_test = X_scaled[:60000], X_scaled[60000:]
    y_train, y_test = y[:60000], y[60000:]

    ann = HandMadeNeuralNetwork(
        input_nodes=784,
        hidden_nodes=100,
        output_nodes=10,
        learning_rate=0.1,
        epochs=5
    )
    ann.train(X_train, y_train)
    predictions = ann.predict(X_test)
    ann_accuracy = accuracy(predictions, y_test)
    print(ann_accuracy)
    input_weights = ann.input_weights
    hidden_weights = ann.hidden_weights

    another = HandMadeNeuralNetwork(
        input_nodes=784,
        hidden_nodes=100,
        output_nodes=10,
        learning_rate=0.1,
        epochs=5,
        input_weights=input_weights,
        hidden_weights=hidden_weights
    )
    predictions = another.predict(X_test)
    another_accuracy = accuracy(predictions, y_test)
    print(another_accuracy)