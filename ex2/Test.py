'''

#####################TEST FOR HYPER PARMETERS#####################
    import sys
    import UTL as UTL
    from Perceptron import Perceptron
    from SVM import SVM
    from PA import PA

    # get the training data
    train_x = UTL.readTrainData('train_x.txt')
    # get the training labels
    train_y = UTL.readTrainY('train_y.txt')
    # get the test data
    test_x = UTL.readTrainData('test_x.txt')

    #normlaize test and train
    train_x, mean, std = UTL.normalize_z_train(train_x)
    test_x = UTL.normalize_z_test(test_x, mean, std)
    #normlaize with min max
    train_x, mean, std = UTL.maxMixNormalize_train(train_x)
    test_x = UTL.maxMixNormalize_test(test_x, mean, std)


    ################## Run all train by LR ######################

    #create eta vector:
    Lr = [i for i in range(0.001,1.5)]

    for i in range(len(Lr)):
        perceptron = Perceptron(train_x, train_y, test_x, eta= Lr[i])
        perceptron.train()
        predictionsPerceptron = perceptron.predict()

        svm = SVM(train_x, train_y, test_x, eta=Lr[i], lamda=0.2)
        svm.train()
        predictionsSVM = svm.predict()

        pa = PA(train_x, train_y, test_x)
        pa.train()
        predictionsPA = pa.predict()

        test_y = UTL.readTrainY("test_y.txt")
        count_svm = 0
        count_pers = 0
        count_pa = 0
        for i in range(len(test_y)):
            if test_y[i] == predictionsSVM[i]:
                count_svm += 1
            if test_y[i] == predictionsPerceptron[i]:
                count_pers += 1
            if test_y[i] == predictionsPA[i]:
                count_pa += 1
        print(count_pers/i,count_svm/i,count_pa/i)

    ################## Run all train by Epoches ######################
    #create ephocs vector:
    ephocs = [10,50,100,200,500,1000]
    for i in range(len(Lr)):
        perceptron = Perceptron(train_x, train_y, test_x,,ephocs=ephocs[i] , eta= 0.1)
        perceptron.train()
        predictionsPerceptron = perceptron.predict()

        svm = SVM(train_x, train_y, test_x,ephocs=ephocs[i] ,eta=0.1, lamda=0.2)
        svm.train()
        predictionsSVM = svm.predict()

        pa = PA(train_x, train_y, test_x)
        pa.train()
        predictionsPA = pa.predict()

        test_y = UTL.readTrainY("test_y.txt")
        count_svm = 0
        count_pers = 0
        count_pa = 0
        for i in range(len(test_y)):
            if test_y[i] == predictionsSVM[i]:
                count_svm += 1
            if test_y[i] == predictionsPerceptron[i]:
                count_pers += 1
            if test_y[i] == predictionsPA[i]:
                count_pa += 1
        print(count_pers/i,count_svm/i,count_pa/i)

#################################################################
'''