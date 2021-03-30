from builtins import object

import numpy as np

class DualTensor(object):
    # Class object for dual representation of a tensor/matrix/vector
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def zero_grad(self):
        # Reset the gradient for the next batch evaluation
        dual_part = np.zeros((len(self.real), len(self.real)))
        np.fill_diagonal(dual_part, 1)
        self.dual = dual_part
        return

def dot_product(b_dual, x, both_require_grad=False):
    # Function to perform dot product dual . grad_req vector
    real_part = np.dot(x.real, b_dual.real)
    dual_part = np.dot(x.real, b_dual.dual)
    if both_require_grad:
        dual_part += np.dot(b_dual.real, x.dual)
    return DualTensor(real_part, dual_part)

def add_duals(dual_a, dual_b):
    # Operator non-"overload": Add a two dual numbers
    real_part = dual_a.real + dual_b.real
    dual_part = dual_a.dual + dual_b.dual
    return DualTensor(real_part, dual_part)

def log(dual_tensor):
    # Operator non-"overload": Log (real) & its derivative (dual)
    real_part = np.log(dual_tensor.real)
    temp_1 = 1 / dual_tensor.real
    # Fill matrix with diagonal entries of log derivative
    temp_2 = np.zeros((temp_1.shape[0], temp_1.shape[0]))
    np.fill_diagonal(temp_2, temp_1)
    dual_part = np.dot(temp_2, dual_tensor.dual)
    return DualTensor(real_part, dual_part)

def sigmoid(dual_tensor):
    # Operator non-"overload": Sigmoid (real) & its derivative (dual)
    real_part = 1/(1+np.exp(-dual_tensor.real))
    temp_1 = np.multiply(real_part, 1-real_part)
    # Fill matrix with diagonal entries of sigmoid derivative
    temp_2 = np.zeros((temp_1.shape[0], temp_1.shape[0]))
    np.fill_diagonal(temp_2, temp_1)
    dual_part = np.dot(temp_2, dual_tensor.dual)
    return DualTensor(real_part, dual_part)

def forward(X, b_dual):
    # Apply element-wise sigmoid activation
    y_pred_1 = sigmoid(dot_product(b_dual, X))
    y_pred_2 = DualTensor(1-y_pred_1.real, -y_pred_1.dual)

    # Clip the bounds
    y_pred_1.real = np.clip(y_pred_1.real, 1e-15, 1 - 1e-15)
    y_pred_2.real = np.clip(y_pred_2.real, 1e-15, 1 - 1e-15)
    return y_pred_1, y_pred_2

def binary_cross_entropy_dual(y_true, y_pred_1, y_pred_2):
    # Compute actual binary cross-entropy term
    log_y_pred_1, log_y_pred_2 = log(y_pred_1), log(y_pred_2)
    bce_l1, bce_l2 = dot_product(log_y_pred_1, -y_true), dot_product(log_y_pred_2, -(1 - y_true))
    bce = add_duals(bce_l1, bce_l2)
    # Calculate the batch classification accuracy
    acc = (y_true == (y_pred_1.real > 0.5)).sum()/y_true.shape[0]
    return bce, acc