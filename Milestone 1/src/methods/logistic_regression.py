import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        # Initialize weights
        self.w = np.linalg.inv(training_data.T @ training_data) @ training_data.T @ label_to_onehot(training_labels, C)
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        """
        y^(k) = (exp((w*_k)^T x))/(sum_j^C exp((w*_j)^T x)))
        Then the label is K = argmax_j y^(j)(x)
        """
        if self.w is None:
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        # print("w shape: ", self.w.shape)
        # print("test data shape: ", test_data.shape)
        A = np.exp(self.w.T @ test_data.T)
        B = np.sum(A, axis=0)
        y_hat = A / B
        pred_labels = np.argmax(y_hat, axis=0)
        return pred_labels
