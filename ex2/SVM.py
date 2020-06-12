import numpy as np
from Algo import Algo

'''
preceptron class
'''
class SVM(Algo):
    def train(self):
            self.w = np.zeros((3, len(self.train_x[0])))
            for j in range(self.epochs):
                for a in range(len(self.train_x)):
                    xi = self.train_x[a][:8]
                    trueClass = self.train_y[a]
                    y_hat = np.argmax(np.dot(self.w, xi))
                    if trueClass != y_hat:
                        for i in range(3):
                            if i == trueClass:
                                self.w[int(trueClass), :] = (1 - self.eta * self.lamda) * self.w[int(trueClass), :] + self.eta * xi
                            elif i == y_hat:
                                self.w[int(y_hat), :] = (1 - self.eta * self.lamda) * self.w[int(y_hat), :] - self.eta * xi
                            else:
                                self.w[i, :] = (1 - self.eta * self.lamda) * self.w[int(i), :]
            self.eta /= j + 1