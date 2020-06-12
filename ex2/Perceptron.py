import numpy as np
from Algo import Algo

'''
preceptron class
'''
class Perceptron(Algo):
    def train(self):
        self.w = np.zeros((3, len(self.train_x[0])))
        for j in range(self.epochs):
            for i in range(len(self.train_x)):
                xi = self.train_x[i][:8]
                trueClass = self.train_y[i]
                y_hat = np.argmax(np.dot(self.w, xi))
                if trueClass != y_hat:
                    self.w[int(trueClass), :] = self.w[int(trueClass), :] + self.eta * xi
                    self.w[int(y_hat), :] = self.w[int(y_hat), :] - self.eta * xi
        self.eta /= j + 1
