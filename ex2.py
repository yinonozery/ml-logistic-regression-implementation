# - Yinon Ozery, ID: 205954621

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


def log_reg_sklearn(X_train, x_test, Y_train, y_test):
    log_regression = LogisticRegression(random_state=42)
    log_regression.fit(X_train, Y_train)
    y_predict = log_regression.predict(x_test)

    print('Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
    print('Accuracy:', str(accuracy_score(y_test, y_predict) * 100)[:5] + '%')
    print('Precision:', precision_score(y_test, y_predict, average=None))
    print('Recall:', recall_score(y_test, y_predict, average=None))
    print('F1 Score:', f1_score(y_test, y_predict, average=None))

# -- Plots -- #
    colorsMap = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

# Training plot
    plt.figure('Actual values')
    for i in range(len(X_train)):
        plt.scatter(X_train[i][1], X_train[i][2],
                    c=colorsMap.get(Y_train[i]))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show(block=False)

# Test plot
    X_train_cols = X_train[:, 1:3]  # Choosing the first 2 columns for the plot
    plt.figure('Predicted values + Decision Boundaries Lines')
    for i in range(len(x_test)):
        plt.scatter(x_test[i][1], x_test[i][2], c=colorsMap.get(y_test[i]))

    # Decision Boundaries
    coeff = log_regression.coef_
    num_of_classes = len(log_regression.classes_)
    num_of_classes = num_of_classes - 1 if num_of_classes == 2 else num_of_classes
    for i in range(0, num_of_classes):
        yd = -(coeff[i][0] + (coeff[i][1]) *
               X_train_cols[:, 0]) / (coeff[i][2])
        plt.plot(X_train_cols[:, 0], yd, label=colorsMap[i] + " vs rest")
    plt.legend()

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def get_thetas(x, y):
    m = len(x)  # num of data examples
    n = x.shape[1]  # num of features
    alpha = 1.0e-2  # learning rate
    ERR = 1.0e-4  # epsilon - convergence "error" for cost function
    max_iterations = 1.0e+3  # max num of iterations
    thetas = np.random.rand(n, 1)  # initilize of Thetas
    thetas_history = thetas.tolist()
    cost_history = []  # storage vector for cost function values
    last_cost = ERR + 1
    dcost = ERR + 1
    i = 0

    # Sigmoid curve
    def g(z):
        p = 1 / (1 + np.exp(-z))
        p = np.minimum(p, 0.9999999999)
        p = np.maximum(p, 0.0000000001)
        return p

    # H(0)
    def h_theta(x):
        return g(thetas.transpose() @ x)

    # J(0)
    def cost():
        cost = 0
        for j in range(0, m):
            cost += y[j][0]*np.log(h_theta(x[j])) + \
                (1 - y[j][0])*np.log(1 - h_theta(x[j]))
        return -cost / m

    while dcost > ERR and i < max_iterations:
        i += 1
        thetas = thetas - (alpha/m) * (x.transpose() @ (g(x @ thetas) - y))
        cost_history.append((i, last_cost))

        # diff of Cost function between the iterations
        curr_cost = cost()[0]

        dcost = abs(curr_cost - last_cost)

        # update lastCost
        last_cost = curr_cost

        # storage vector of thetas
        for j in range(0, n):
            thetas_history.append(thetas)

    return thetas


def log_reg_ovr(num_of_classes, X_train, x_test, Y_train, y_test, y):
    all_thetas = []
    training_len = len(X_train)
    test_len = len(x_test)

    # "Merging" classes (one-Vs-rest)
    for i in range(0, num_of_classes):
        Y_others = np.ndarray((training_len, 1)).astype(int)
        for j in range(0, training_len):
            if Y_train[j] == i:
                Y_others[j] = 1
            else:
                Y_others[j] = 0
        all_thetas.append(get_thetas(X_train, Y_others))

    # Predict values of x_test
    maxProb = []
    for i in range(0, test_len):
        maxProb.append((-1, -1))

    for i in range(0, num_of_classes):
        newProb = (1 / (1 + np.exp(-(x_test @ all_thetas[i]))))
        for j in range(0, test_len):
            if maxProb[j][0] < newProb[j]:
                maxProb[j] = (newProb[j], i)

    y_predict = []
    [y_predict.append(x[1]) for x in maxProb]

# -- Prints -- #
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
    print('Accuracy:', str(accuracy_score(y_test, y_predict) * 100)[:5] + '%')
    print('Precision:', precision_score(y_test, y_predict, average=None))
    print('Recall:', recall_score(y_test, y_predict, average=None))
    print('F1 Score:', f1_score(y_test, y_predict, average=None))

# -- Plots -- #
    colorsMap = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

    # Training plot
    x1 = X_train[:, 1]
    x2 = X_train[:, 2]
    plt.figure('Actual values')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(0, len(X_train)):
        if y[i] == 0:
            plt.scatter(x1[i], x2[i], c=colorsMap.get(Y_train[i]))
        else:
            plt.scatter(x1[i], x2[i], c=colorsMap.get(Y_train[i]))

    plt.show(block=False)

    # Test plot
    x1 = x_test[:, 1]
    x2 = x_test[:, 2]
    plt.figure('Predicted values + Decision Boundaries Lines')
    plt.xlabel('x1')
    plt.ylabel('x2')

    for i in range(0, len(y_predict)):
        if y[i] == 0:
            plt.scatter(x1[i], x2[i], c=colorsMap.get(y_predict[i]))
        else:
            plt.scatter(x1[i], x2[i], c=colorsMap.get(y_predict[i]))

    # Decision Boundaries
    X_train_cols = x_test[:, 1:3]
    num_of_classes = num_of_classes - 1 if num_of_classes == 2 else num_of_classes
    for i in range(0, num_of_classes):
        yd = -(all_thetas[i][0] + (all_thetas[i][1]) *
               X_train_cols[:, 0]) / (all_thetas[i][2])
        plt.plot(x1, yd, label=colorsMap[i] + " vs rest")
    plt.legend()

    plt.show()


def get_training_data(_n_classes, _n_samples):
    x, y = make_classification(n_samples=_n_samples,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_classes=_n_classes,
                               n_clusters_per_class=1,
                               class_sep=2,
                               random_state=42)
    x = np.insert(x, 0, [1] * _n_samples, axis=1)  # insert column of one's
    y = np.array(y.reshape(-1, ))  # shape (n_samples, )
    return x, y


def main():
    # Get the number of samples and classes
    num_of_classes = 0
    num_of_samples = int(input("Number of random samples: "))
    while (num_of_classes < 2 or num_of_classes > 4):
        num_of_classes = int(input("Choose the number of classes 2/3/4: "))

    # Get random data samples
    x, y = get_training_data(num_of_classes, num_of_samples)
    X_train, x_test, Y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # Linear Regression - Sklearn Method
    log_reg_sklearn(X_train, x_test, Y_train, y_test)

    # Linear Regression - Manual Method (One-vs-Rest)
    log_reg_ovr(num_of_classes, X_train, x_test, Y_train, y_test, y)


if __name__ == '__main__':
    main()
