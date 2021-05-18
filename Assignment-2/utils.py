import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

colors = ['navy', 'darkorange']


def load_data(train_file, test_file, delimiter=','):
    train_data = np.loadtxt(train_file, delimiter=delimiter)
    test_data = np.loadtxt(test_file, delimiter=delimiter)
    data = {}
    data["train"] = train_data[:, :-1]
    data["train_labels"] = train_data[:, -1]
    data["test"] = {'X': test_data[:, :-1], 'y': test_data[:, -1]}
    return data


def get_class_samples(data, n_samples, plot=False):
    class2_indices = np.where(data["train_labels"] == -1)[0]
    class1_indices = np.where(data["train_labels"] == 1)[0]

    n_class2 = class2_indices.shape[0]
    n_class1 = class1_indices.shape[0]

    # Select random indices for n_samples
    class2_select_indices = class2_indices[np.random.choice(n_class2, n_samples, replace=False)]
    class1_select_indices = class1_indices[np.random.choice(n_class1, n_samples, replace=False)]

    class2_samples = data["train"][class2_select_indices]
    class1_samples = data["train"][class1_select_indices]

    train_samples = {}
    train_samples["class1"] = class1_samples
    train_samples["class2"] = class2_samples

    if plot:
        plt.figure()
        plt.scatter(class1_samples[:, 0], class1_samples[:, 1], marker='+', c='r', label='Class1')
        plt.scatter(class2_samples[:, 0], class2_samples[:, 1], label='Class2')
        plt.title(f'Samples per class = {n_samples}')
        plt.legend(loc="best")
        plt.show()

    knn_data = {}
    knn_data["X"] = np.append(train_samples["class1"], train_samples["class2"], axis=0)
    knn_data["y"] = np.append(data["train_labels"][class1_select_indices], data["train_labels"][class2_select_indices],
                              axis=0)

    sampled_data = {}
    sampled_data["knn"] = knn_data
    sampled_data["bayes"] = train_samples
    return sampled_data


def linear_classifier(train_data, test_data, classifiers=['linear', 'logistic'], plot=False):
    X = train_data['X']
    Y = train_data['y']
    n_samples = Y.shape[0] / 2

    if plot:
        plt.figure()
        class1_data = X[np.where(Y == 1)]
        class2_data = X[np.where(Y == -1)]
        class1_x, class1_y = class1_data[:, 0], class1_data[:, 1]
        class2_x, class2_y = class2_data[:, 0], class2_data[:, 1]

        plt.scatter(class1_x, class1_y, marker='+', c=colors[0], label='Class1')
        plt.scatter(class2_x, class2_y, marker='+', c=colors[1], label='Class2')

        x_min = min(class1_x.min(), class2_x.min())
        y_min = min(class1_y.min(), class2_y.min())
        x_max = max(class1_x.max(), class2_x.max())
        y_max = max(class1_y.max(), class2_y.max())

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        x_vals = np.linspace(x_min, x_max, 10)
        # plt.show()

    accuracy = {}
    for clf_type in classifiers:
        if clf_type == 'linear':
            # exact soln
            A = np.ones((X.shape[0], X.shape[1] + 1))
            A[:, 1:] = X
            W_exact = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), np.matmul(A.T, Y))
            #print(W_exact)

            # clf = make_pipeline(StandardScaler(), linear_model.LogisticRegression())
            clf = linear_model.LinearRegression()
            clf.fit(X, Y)
            W = clf.coef_
            w1 = W[0]
            w2 = W[1]
            w0 = clf.intercept_
            W_lib = np.append(np.array([w0]), W)

            # print(f'Formula est W:{W_exact}')
            # print(f'Lib estimated W:{W_lib}')
            assert np.allclose(W_lib, W_exact)
            logits = clf.predict(test_data['X'])
            preds = np.zeros(logits.shape)
            preds[np.where(logits > 0)] = 1
            preds[np.where(logits <= 0)] = -1
            metrics = confusion_matrix(test_data['y'], preds)
            acc = np.trace(metrics) / np.sum(metrics)
            accuracy[clf_type] = acc
            class_acc = [metrics[0, 0] / 20 * 100, metrics[1, 1] / 10 * 100]
            print(class_acc, acc)

            if plot:
                y_vals = -(w0 + w1 * x_vals) / w2
                plt.plot(x_vals, y_vals, '--', c='red', label='linear')
                plt.title(f'n_samples = {int(n_samples)}')
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.legend(loc="best")

        if clf_type == 'logistic':
            # clf = make_pipeline(StandardScaler(), linear_model.LogisticRegression(penalty='none'))
            Y_logistic = deepcopy(Y)
            #Y_logistic[np.where(Y==-1)]=0
            clf = linear_model.LogisticRegression(penalty='none', multi_class='ovr')
            clf.fit(X, Y_logistic)
            #print(clf.coef_,clf.intercept_)
            W = clf.coef_.T
            #print(W)
            w1 = W[0]
            w2 = W[1]
            w0 = clf.intercept_
            test_y_logistic = deepcopy(test_data['y'])
            #test_y_logistic[np.where(test_data['y'] == -1)] = 0
            accuracy[clf_type] = clf.score(test_data['X'], test_y_logistic)
            #print(clf.predict(test_data['X']))

            if plot:
                y_vals = -(w0 + w1 * x_vals) / w2
                plt.plot(x_vals, y_vals, '--', c='green', label = 'logistic')
                plt.title(f'n_samples = {int(n_samples)}')
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.legend(loc="best")

        # print(f'Confusion matrix:\n {metrics}')
        # print(f'Accuracy:{acc} for {n_samples} samples')

        if False:
            plt.figure()
            class1_data = X[np.where(Y == 1)]
            class2_data = X[np.where(Y == -1)]
            class1_x, class1_y = class1_data[:, 0], class1_data[:, 1]
            class2_x, class2_y = class2_data[:, 0], class2_data[:, 1]

            plt.scatter(class1_x, class1_y, marker='+', c=colors[0], label='Class1')
            plt.scatter(class2_x, class2_y, marker='+', c=colors[1], label='Class2')

            x_min = min(class1_x.min(), class2_x.min())
            y_min = min(class1_y.min(), class2_y.min())
            x_max = max(class1_x.max(), class2_x.max())
            y_max = max(class1_y.max(), class2_y.max())

            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            x_vals = np.linspace(x_min, x_max, 10)
            y_vals = -(w0 + w1 * x_vals) / w2
            plt.plot(x_vals, y_vals, '--', c='red')
            plt.title(f'n_samples = {int(n_samples)}')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend(loc="best")
    plt.show()
    return accuracy


def linear_least_squares_3class(train_data, test_data, plot=False):
    X = train_data['X']
    Y = train_data['y']
    Y_onehot = np.eye(3)[Y.astype(np.uint8)]
    n_samples = Y.shape[0] / 2

    A = np.ones((X.shape[0], X.shape[1] + 1))
    A[:, 1:] = X
    W_exact = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), np.matmul(A.T, Y_onehot))
    print(W_exact.shape)

    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(X, Y_onehot)
    W = linear_reg.coef_
    w1 = W[0]
    w2 = W[1]
    w0 = linear_reg.intercept_
    W_lib = np.zeros((3, 5))
    W_lib[:, 0] = w0
    W_lib[:, 1:] = W
    W_lib = W_lib.T

    assert np.allclose(W_lib, W_exact)
    # print(f'Formula est W:{W_exact}')
    # print(f'Lib estimated W:{W_lib}')

    logits = linear_reg.predict(test_data['X'])
    preds = np.argmax(logits, axis=1)
    metrics = confusion_matrix(test_data['y'], preds)
    accuracy = np.trace(metrics) / np.sum(metrics)
    class_acc = [metrics[0,0]/10*100, metrics[1,1]/10*100, metrics[2,2]/10*100]
    print(class_acc, accuracy)
    # print(f'Confusion matrix: {metrics}\n\n')
    # print(f'Accuracy:{accuracy} for {n_samples} samples')

    if plot:
        plt.figure()
        class1_data = X[np.where(Y == 1)]
        class2_data = X[np.where(Y == -1)]
        class1_x, class1_y = class1_data[:, 0], class1_data[:, 1]
        class2_x, class2_y = class2_data[:, 0], class2_data[:, 1]

        plt.scatter(class1_x, class1_y, marker='+', c=colors[0], label='Class1')
        plt.scatter(class2_x, class2_y, marker='+', c=colors[1], label='Class2')

        x_min = min(class1_x.min(), class2_x.min())
        y_min = min(class1_y.min(), class2_y.min())
        x_max = max(class1_x.max(), class2_x.max())
        y_max = max(class1_y.max(), class2_y.max())

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        x_vals = np.linspace(x_min, x_max, 10)
        y_vals = -(w0 + w1 * x_vals) / w2
        plt.plot(x_vals, y_vals, '--', c='red')
        plt.title(f'Samples per class = {int(n_samples)}')
        plt.legend(loc="best")
        plt.show()
    return accuracy
