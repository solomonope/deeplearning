import numpy as np;


class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights);
        self.bias = bias;

    def predict(self, input):
        return np.dot(self.weights.T, np.array(input)) + self.bias


class Sigmoid(Perceptron):
    def predict(self, input):
        x = super(Sigmoid, self).predict(input)
        return 1 / (1 + np.exp(-x));


if __name__ == "__main__":
    perceptron = Perceptron([2, 2], -4);
    sigmoid = Sigmoid([2*30000, 2*30000], -4*30000);
    print(perceptron.predict([5, 6]))
    print(sigmoid.predict([5, 6]))



#18
#1.52299795128e-08