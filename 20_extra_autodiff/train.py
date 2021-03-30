from Dataloader import DataLoader
from sklearn.linear_model import LogisticRegression
from DualTensor import *
import matplotlib.pyplot as plt

import numpy as np

def train_logistic_regression(n, d, n_epoch, batch_size, b_init, l_rate):
    # Generate the data for a coefficient vector & init progress tracker!
    data_loader = DataLoader(n, d, batch_size, binary=True)
    b_hist, func_val_hist, param_error, acc_hist = [], [], [], []

    # Get the coefficients as solution to optimized sklearn function
    logreg = LogisticRegression(penalty='none', solver='lbfgs', multi_class='multinomial')
    logreg.fit(data_loader.X, data_loader.y)
    norm_coeff = np.linalg.norm(logreg.coef_.ravel())

    b_dual = DualTensor(b_init, None)

    for epoch in range(n_epoch):
        # Shuffle the batch identities at beginning of each epoch
        data_loader.batch_shuffle()
        for batch_id in range(data_loader.num_batches):
            # Clear the gradient
            b_dual.zero_grad()

            # Select the current batch & perform "mini-forward" pass
            X, y = data_loader.get_batch_idx(batch_id)
            y_pred_1, y_pred_2 = forward(X, b_dual)

            # Calculate the forward AD - real = func, dual = deriv
            current_dual, acc = binary_cross_entropy_dual(y, y_pred_1, y_pred_2)

            # Perform grad step & append results to the placeholder list
            b_dual.real -= l_rate * np.array(current_dual.dual).flatten()
            b_hist.append(b_dual.real)

            func_val_hist.append(current_dual.real)

            param_error.append(np.linalg.norm(logreg.coef_.ravel() - b_hist[-1]) / norm_coeff)
            acc_hist.append(acc)

        if np.abs(param_error[-1] - param_error[-2]) < 0.00001:
            break

        if epoch % 1 == 0:
            print(
                "Accuracy: {} | Euclidean Param Norm: {} | fct min: {}".format(acc, param_error[-1], current_dual.real))
    return b_hist, func_val_hist, param_error, acc_hist

if __name__ == "__main__":
    np.random.seed(1)
    b, f, error, acc = train_logistic_regression(1000, 4, 10, 32, np.array([0, 0, 0, 0, 0]).astype(float), 0.001)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(f)
    plt.ylabel("Batch Cross-Entropy Loss", fontsize=15)
    # plt.ylim(bottom=-10, top=450)
    plt.xlabel(r"Batch Iterations", fontsize=15)
    plt.title("Binary Cross-Entropy Loss", fontsize=16)

    plt.subplot(1, 3, 2)
    plt.plot(error)
    plt.ylabel(r"$||\beta^{ad} - \beta^{sklearn}||/||\beta^{sklearn}||$", fontsize=15)
    # plt.ylim(bottom=-10, top=450)
    plt.xlabel(r"Batch Iterations", fontsize=15)
    plt.title(r"Convergence to $\beta^{sklearn}$", fontsize=16)

    plt.subplot(1, 3, 3)
    plt.plot(acc)
    plt.ylabel(r"Batch Accuracy", fontsize=15)
    # plt.ylim(bottom=-10, top=450)
    plt.xlabel(r"Batch Iterations", fontsize=15)
    plt.title("Train Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig("logistic_regression.png", dpi=300)