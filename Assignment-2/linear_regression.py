import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from utils import load_data, get_class_samples, linear_classifier, linear_least_squares_3class
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

import seaborn as sns
import pandas as pd

colors = ['navy', 'darkorange']


def problem1():  # Linear least squares vs Logistic Regression

    n_runs = 10

    acc_dict = {'10': 0, '50': 0, '100': 0, '500': 0, '1000': 0}

    # Gamma distribution
    print('Linear Least squares for 2D Gamma distributed data')
    p1a_data = load_data('Gamma_train.txt', 'Gamma_test.txt')
    p1a_acc_linear = deepcopy(acc_dict)
    p1a_acc_logistic = deepcopy(acc_dict)
    for n_samples in [10, 50, 100, 500, 1000]:
        acc_linear = 0
        acc_logistic = 0
        for i in range(n_runs):
            sampled_data = get_class_samples(p1a_data, n_samples)
            train_data = sampled_data["knn"]
            acc = linear_classifier(train_data, p1a_data["test"], plot=False)
            acc_linear += acc['linear']
            acc_logistic += acc['logistic']
        p1a_acc_linear[str(n_samples)] = np.round(acc_linear / n_runs * 100, 2)
        p1a_acc_logistic[str(n_samples)] = np.round(acc_logistic / n_runs * 100, 2)
    print(f'Accuracy of Linear regression models:\n {p1a_acc_linear}')
    print(f'Accuracy of Logistic regression models:\n {p1a_acc_logistic}')

    # plt.plot(*zip(*acc_gamma.items()))
    # plt.ylim((0,100))
    # plt.show()

    print('\n\nLinear Least squares for 2D uniform distributed data')
    p1b_data = load_data('Uniform_train.txt', 'Uniform_test.txt')
    p1b_acc_linear = deepcopy(acc_dict)
    p1b_acc_logistic = deepcopy(acc_dict)
    for n_samples in [10, 50, 100, 500, 1000]:
        acc_linear = []
        acc_logistic = []
        for i in range(n_runs):
            sampled_data = get_class_samples(p1b_data, n_samples)
            train_data = sampled_data["knn"]
            acc = linear_classifier(train_data, p1b_data["test"], plot=False)
            acc_linear.append(acc['linear'])
            acc_logistic.append(acc['logistic'])
        p1b_acc_linear[str(n_samples)] = np.round(np.mean(acc_linear) * 100, 2)
        p1b_acc_logistic[str(n_samples)] = np.round(np.mean(acc_logistic)  * 100, 2)
    print(f'Accuracy of Linear regression models:\n {p1b_acc_linear}')
    print(f'Accuracy of Logistic regression models:\n {p1b_acc_logistic}')

    print('\n\nLinear Least squares for 10D Normal distributed data')
    p1c_data = load_data('Normal_train_10D.txt', 'Normal_test_10D.txt')
    p1c_acc_linear = deepcopy(acc_dict)
    p1c_acc_logistic = deepcopy(acc_dict)
    for n_samples in [10, 50, 100, 500, 1000]:
        acc_linear = []
        acc_logistic = []
        for i in range(n_runs):
            sampled_data = get_class_samples(p1c_data, n_samples)
            train_data = sampled_data["knn"]
            acc = linear_classifier(train_data, p1c_data["test"], plot=False)
            acc_linear.append(acc['linear'])
            acc_logistic.append(acc['logistic'])
        p1c_acc_linear[str(n_samples)] = np.round(np.mean(acc_linear)  * 100, 2)
        p1c_acc_logistic[str(n_samples)] = np.round(np.mean(acc_logistic)  * 100, 2)
    print(f'Accuracy of Linear regression models:\n {p1c_acc_linear}')
    print(f'Accuracy of Logistic regression models:\n {p1c_acc_logistic}')

    return


def problem2():
    print('\n\nLinear Least squares for 4D Iris dataset\n\n')
    data = np.loadtxt('iris_dataset.txt', delimiter=',')
    data_X = data[:, :-1]
    data_y = data[:, -1] - 1  # convert indices from [1,2,3] to [0,1,2]

    test_class0 = np.random.choice(50, 10, replace=False)
    test_class1 = np.random.choice(50, 10, replace=False) + 50
    test_class2 = np.random.choice(50, 10, replace=False) + 100
    test_indices = np.append(np.append(test_class0, test_class1), test_class2)
    test_data = {'X': data_X[test_indices], 'y': data_y[test_indices]}

    train_indices = np.setdiff1d(np.arange(0, 150), test_indices)
    train_data = {'X': data_X[train_indices], 'y': data_y[train_indices]}

    print('Testing on IRIS dataset...')
    acc_3class_linear = linear_least_squares_3class(train_data, test_data)
    print(f'Accuracy of 3 class linear least squares model = {acc_3class_linear:.2f}\n\n')

    # Class1 VS (2,3)
    train_1vs23 = deepcopy(train_data)
    train_1vs23['y'][np.where(train_1vs23['y'] != 0)] = -1
    train_1vs23['y'][np.where(train_1vs23['y'] == 0)] = 1

    test_1vs23 = deepcopy(test_data)
    test_1vs23['y'][np.where(test_1vs23['y'] != 0)] = -1
    test_1vs23['y'][np.where(test_1vs23['y'] == 0)] = 1
    acc_1vs23_linear = linear_classifier(train_1vs23, test_1vs23, classifiers=['linear'])
    print(f'Accuracy of Class 1 Vs (2,3) = {acc_1vs23_linear}\n\n')

    # Class2 VS (1,3)
    train_2vs13 = deepcopy(train_data)
    train_2vs13['y'][np.where(train_2vs13['y'] != 1)] = -1

    test_2vs13 = deepcopy(test_data)
    test_2vs13['y'][np.where(test_2vs13['y'] != 1)] = -1
    acc_2vs13_linear = linear_classifier(train_2vs13, test_2vs13, classifiers=['linear'])
    print(f'Accuracy of Class 2 Vs (1,3) = {acc_2vs13_linear}\n\n')

    # Class3 VS (1,2)
    train_3vs12 = deepcopy(train_data)
    train_3vs12['y'][np.where(train_3vs12['y'] != 2)] = -1
    train_3vs12['y'][np.where(train_3vs12['y'] == 2)] = 1

    test_3vs12 = deepcopy(test_data)
    test_3vs12['y'][np.where(test_3vs12['y'] != 2)] = -1
    test_3vs12['y'][np.where(test_3vs12['y'] == 2)] = 1
    acc_3vs12_linear = linear_classifier(train_3vs12, test_3vs12, classifiers=['linear'])
    print(f'Accuracy of Class 3 Vs (1,2) = {acc_3vs12_linear}')

    # dataframe = pd.DataFrame.from_records(data, columns=['x0', 'x1', 'x2', 'x3', 'class label'])
    # sns.pairplot(dataframe, hue='class label', diag_kind='auto', corner=True)
    # plt.show()

    return


def problem3():
    data = np.loadtxt('german_credit_data.txt')
    data_X = data[:, :-1]
    data_y = data[:, -1]
    data_y[np.where(data_y == 2)] = -1
    print(np.where(data_y==1)[0].shape)
    print(np.where(data_y == -1)[0].shape)

    n_runs = 1

    # dataframe = pd.DataFrame.from_records(data[:, [0, 1, 2, 3, 23]], columns=['f0', 'f1', 'f2', 'f3', 'label'])
    # sns.pairplot(dataframe, hue='label', diag_kind='hist')
    # plt.show()

    pos_linear = 0
    pos_logistic = 0
    neg_linear = 0
    neg_logistic = 0

    pos_linear_norm = 0
    pos_logistic_norm = 0
    neg_linear_norm = 0
    neg_logistic_norm = 0

    acc_linear = 0
    acc_logistic = 0

    acc_linear_norm = 0
    acc_logistic_norm = 0

    for i in range(n_runs):
        test_indices = np.random.choice(1000, 200, replace=False)
        print(f'\n Run = {i}')
        print(np.where(data_y[test_indices] == 1)[0].shape)
        print(np.where(data_y[test_indices] == -1)[0].shape)
        test_data = {'X': data_X[test_indices], 'y': data_y[test_indices]}

        train_indices = np.setdiff1d(np.arange(0, 1000), test_indices)
        train_data = {'X': data_X[train_indices], 'y': data_y[train_indices]}

        linear_clf = linear_model.LinearRegression()
        linear_clf.fit(train_data['X'], train_data['y'])
        logits = linear_clf.predict(test_data['X'])
        preds = np.zeros(logits.shape)
        preds[np.where(logits > 0)] = 1
        preds[np.where(logits <= 0)] = -1
        metrics = confusion_matrix(test_data['y'], preds)
        acc1 = np.trace(metrics) / np.sum(metrics)
        acc_linear += acc1*100
        class_metrics = confusion_matrix(test_data['y'], preds, normalize='true')
        neg_linear +=class_metrics[0, 0]*100
        pos_linear += class_metrics[1, 1]*100
        print(acc_linear, pos_linear, neg_linear)


        linear_clf_norm = make_pipeline(StandardScaler(), linear_model.LinearRegression())
        linear_clf_norm.fit(train_data['X'], train_data['y'])
        logits = linear_clf_norm.predict(test_data['X'])
        preds = np.zeros(logits.shape)
        preds[np.where(logits > 0)] = 1
        preds[np.where(logits <= 0)] = -1
        metrics = confusion_matrix(test_data['y'], preds)
        acc1 = np.trace(metrics) / np.sum(metrics)
        acc_linear_norm += acc1*100
        class_metrics = confusion_matrix(test_data['y'], preds, normalize='true')
        neg_linear_norm +=class_metrics[0, 0]*100
        pos_linear_norm += class_metrics[1, 1]*100
        print(acc_linear_norm, pos_linear_norm, neg_linear_norm)


        logistic_clf = linear_model.LogisticRegression(penalty='none', max_iter=1000) #, class_weight='balanced')
        logistic_clf.fit(train_data['X'], train_data['y'])
        acc2 = logistic_clf.score(test_data['X'], test_data['y'])
        acc_logistic += acc2*100
        logits = logistic_clf.predict(test_data['X'])
        preds = np.zeros(logits.shape)
        preds[np.where(logits > 0)] = 1
        preds[np.where(logits <= 0)] = -1
        class_metrics = confusion_matrix(test_data['y'], preds, normalize='true')
        neg_logistic +=class_metrics[0, 0]*100
        pos_logistic += class_metrics[1, 1]*100
        print(logits)
        print(acc_logistic, pos_logistic, neg_logistic)

        logistic_clf_norm = make_pipeline(StandardScaler(), linear_model.LogisticRegression(penalty='none'))
        logistic_clf_norm.fit(train_data['X'], train_data['y'])
        acc2_norm = logistic_clf_norm.score(test_data['X'], test_data['y'])
        acc_logistic_norm += acc2_norm*100
        logits = logistic_clf_norm.predict(test_data['X'])
        preds = np.zeros(logits.shape)
        preds[np.where(logits > 0)] = 1
        preds[np.where(logits <= 0)] = -1
        class_metrics = confusion_matrix(test_data['y'], preds, normalize='true')
        neg_logistic_norm +=class_metrics[0, 0]*100
        pos_logistic_norm += class_metrics[1, 1]*100
        print(logits)
        print(acc_logistic_norm, pos_logistic_norm, neg_logistic_norm)


    acc_linear /= n_runs
    acc_logistic /= n_runs

    pos_linear /= n_runs
    neg_linear /= n_runs

    pos_logistic /= n_runs
    neg_logistic /= n_runs

    acc_linear_norm /= n_runs
    acc_logistic_norm /= n_runs

    pos_linear_norm /= n_runs
    neg_linear_norm /= n_runs

    pos_logistic_norm /= n_runs
    neg_logistic_norm /= n_runs

    print(f'Accuracy of good credit Linear regression models:\n {pos_linear:.3f}')
    print(f'Accuracy of bad credit Linear regression models:\n {neg_linear:.3f}')
    print(f'Accuracy of Linear regression models:\n {acc_linear:.2f}')


    print(f'\n\nAccuracy of good credit Logistic regression models:\n {pos_logistic:.3f}')
    print(f'Accuracy of bad credit Logistic regression models:\n {neg_logistic:.3f}')
    print(f'Accuracy of Logistic regression models:\n {acc_logistic:.3f}')

    print(f'\n\nAccuracy of good credit norm Linear regression models:\n {pos_linear_norm:.3f}')
    print(f'Accuracy of bad credit norm Linear regression models:\n {neg_linear_norm:.3f}')
    print(f'Accuracy of norm Linear regression models:\n {acc_linear_norm:.2f}')


    print(f'\n\nAccuracy of good credit norm Logistic regression models:\n {pos_logistic_norm:.3f}')
    print(f'Accuracy of bad credit norm Logistic regression models:\n {neg_logistic_norm:.3f}')
    print(f'Accuracy of norm Logistic regression models:\n {acc_logistic_norm:.3f}')


    return


def problem4():
    p4_data = np.loadtxt('1D_regression_data.txt')
    n_runs = 1

    data_X = p4_data[:, 0].reshape(-1, 1)
    data_y = p4_data[:, 1]

    kf = KFold(n_splits=5, shuffle=True)

    test_mse = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
    train_mse = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}

    for i in range(n_runs):
        train_data, test_data, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

    for train_index, val_index in kf.split(data_X):
        train_data, y_train = data_X[train_index], data_y[train_index]
        test_data, y_test = data_X[val_index], data_y[val_index]
        print(train_index)
        print(val_index)


        x_true = np.linspace(-7, 7, 100)
        y_true = 0.25 * (x_true ** 3) + 1.25 * (x_true ** 2) - 3 * x_true - 3
        plt.plot(x_true, y_true, label = 'True', c='black')
        for degree in range(1, 9):
            poly_reg = PolynomialFeatures(degree=degree)
            X_train = poly_reg.fit_transform(train_data.reshape(-1, 1))
            X_test = poly_reg.fit_transform(test_data.reshape(-1, 1))

            model = linear_model.LinearRegression(fit_intercept=False)
            model.fit(X_train, y_train)
            print(f'Degree = {degree}, coeff:{model.coef_}')

            train_pred = model.predict(X_train)
            train_mse[str(degree)] += np.mean((train_pred - y_train) ** 2)

            test_pred = model.predict(X_test)
            test_mse[str(degree)] += np.mean((test_pred - y_test) ** 2)
            plt.plot(x_true, model.predict(poly_reg.fit_transform(x_true.reshape(-1, 1))), label = str(degree), ls = '--')
        plt.scatter(train_data, y_train, marker='o', c='red', label='CV train')
        plt.scatter(test_data, y_test, marker='+', c='red', label='CV val')
        plt.xlim((-7,7))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #plt.title(f'degree = {degree}')
        plt.show()

    train_mse = dict([(int(k), v/ kf.get_n_splits()) for (k, v) in train_mse.items()])
    test_mse = dict([(int(k), v / kf.get_n_splits()) for (k, v) in test_mse.items()])

    print(f'Train MSE for varying degrees:\n {train_mse}')
    print(f'Test MSE for varying degrees:\n {test_mse}')

    plt.plot(train_mse.keys(), train_mse.values(), label='CV Train MSE')
    plt.plot(test_mse.keys(), test_mse.values(), label='CV Val MSE')
    plt.xlabel('degree')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


# problem1()
# problem2()
problem3()
# problem4()
