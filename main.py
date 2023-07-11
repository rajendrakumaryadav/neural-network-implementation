from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot
from numpy import mean, std
from sklearn.model_selection import KFold


def loadDataset():
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
    train_Y = to_categorical(train_Y)
    test_Y = to_categorical(test_Y)
    return train_X, train_Y, test_X, test_Y


def preprocess(trainSet, testSet):
    train_norm = trainSet.astype("float32")
    test_norm = testSet.astype("float32")
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def modelDefinition():
    model = Sequential()
    # define learning rate
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1),
            kernel_initializer="he_uniform",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def modelEvaluation(Xdata, Ydata, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, random_state=1, shuffle=True)
    for train_ix, test_ix in kfold.split(Xdata):
        model = modelDefinition()
        train_X, train_Y, test_X, test_Y = (
            Xdata[train_ix],
            Ydata[train_ix],
            Xdata[test_ix],
            Ydata[test_ix],
        )
        history = model.fit(
            train_X,
            train_Y,
            epochs=10,
            batch_size=32,
            validation_data=(test_X, test_Y),
            verbose=0,
        )
        _, acc = model.evaluate(test_X, test_Y, verbose=0)
        print("> %.3f" % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories


def summary(scores):
    print("Accuracy: mean=%.3f" % (mean(scores) * 100))
    pyplot.boxplot(scores)
    pyplot.show()


def runTest():
    train_X, train_Y, test_X, test_Y = loadDataset()
    train_X, test_X = preprocess(train_X, test_X)
    scores, histories = modelEvaluation(train_X, train_Y)
    summary(scores)


runTest()
