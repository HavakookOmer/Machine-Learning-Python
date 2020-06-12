import numpy as np
'''
class algo
interface for 3 algorithems
'''
class Algo:
    '''
    constructor
    '''
    def __init__(self, data_x, data_y, test_x, weights=None, epochs=10, lamda=None, eta=None):
        self.train_x = data_x
        self.train_y = data_y
        self.test_x = test_x
        self.epochs = epochs
        self.w = weights
        self.eta = eta #Learning rate
        self.lamda = lamda

    '''
    predict function
    '''
    def predict(self):
        predict = []
        for i in range(len(self.test_x)):
            predict.append(np.argmax(np.dot(self.w,self.test_x[i])))
        return predict

