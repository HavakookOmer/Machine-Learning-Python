import sys
import UTL as UTL
from Perceptron import Perceptron
from SVM import SVM
from PA import PA

if __name__ == '__main__':

    # get args from terminal
    arguments = sys.argv[1:]
    train_x_file = arguments[0]  # Train
    train_y_file = arguments[1]  # Labels
    test_x_file = arguments[2]  # Test

    # get the training data
    train_x = UTL.readTrainData(train_x_file)
    # get the training labels
    train_y = UTL.readTrainY(train_y_file)
    # get the test data
    test_x = UTL.readTrainData(test_x_file)

    #normlaize test and train
    train_x, mean, std = UTL.normalize_z_train(train_x)
    test_x = UTL.normalize_z_test(test_x, mean, std)

    #Run all train and print predictions
    perceptron = Perceptron(train_x, train_y, test_x, eta= 0.1)
    perceptron.train()
    predictionsPerceptron = perceptron.predict()

    svm = SVM(train_x, train_y, test_x, eta=0.015, lamda=0.2)
    svm.train()
    predictionsSVM = svm.predict()

    pa = PA(train_x, train_y, test_x)
    pa.train()
    predictionsPA = pa.predict()

    for a,b,c in zip(predictionsPerceptron,predictionsSVM,predictionsPA):
        print("perceptron: {}, svm: {}, pa: {}".format(a,b,c))
