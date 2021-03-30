from builtins import len

import numpy as np

class DataLoader(object):
    def __init__(self, n, d, batch_size, binary=False):
        self.total_dim = d+1
        self.X, self.y = self.generate_regression_data(n, d, binary)
        # Set batch_id
        self.num_batches = np.ceil(n/batch_size).astype(int)
        self.batch_ids = np.array([np.repeat(i, batch_size) for i in range(self.num_batches)]).flatten()[:n]

    def generate_regression_data(self, n, d, binary=False):
        self.b_true = self.generate_coefficients(d)
        X = np.random.normal(0, 1, n*d).reshape((n, d))
        noise = np.random.normal(0, 1, n).reshape((n, 1))
        inter = np.ones(n).reshape((n, 1))
        X = np.hstack((inter, X))
        y = np.matmul(X, self.b_true) + noise
        # Make data binary if task is classification/logistic regression
        if binary:
            y[y > 0] = 1
            y[y <= 0] = 0
        return X, y

    def generate_coefficients(self, d, intercept=True):
        # Generate random integer-valued coefficients for Xb + e
        b_random = np.random.randint(-5, 5, d + intercept)
        return b_random.reshape((d + intercept, 1))

    def batch_shuffle(self):
        # Diversity in the minibatch
        assert len(self.X) == len(self.y)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def get_batch_idx(self, batch_id):
        # Subselect the current batch to be processed!
        idx = np.where(self.batch_ids == batch_id)[0]
        return self.X[idx, :], self.y[idx].flatten()
