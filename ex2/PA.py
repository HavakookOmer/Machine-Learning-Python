import numpy as np
from Algo import Algo

'''
Passive Aggresive class
'''
class PA(Algo):
    def train(self):
        self.w = np.zeros((3, len(self.train_x[0])))
        for j in range(self.epochs):
            for i in range(len(self.train_x)):
                xi = self.train_x[i][:8]
                trueClass = self.train_y[i]
                y_hat = int(np.argmax(np.dot(self.w, xi)))
                l_value = max(0, (1 - (np.dot(self.w[int(trueClass)], xi)) + (np.dot(self.w[int(y_hat)], xi))))
                norm_x = (np.linalg.norm(xi)) ** 2
                if norm_x == 0:
                    tau = 0
                else:
                    tau = l_value / (2 * norm_x)
                if trueClass != y_hat:
                    self.w[int(trueClass), :] = self.w[int(trueClass), :] + tau * xi
                    self.w[int(y_hat), :] = self.w[int(y_hat), :] - tau * xi
