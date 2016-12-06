import numpy as np
import math
import random
'''
    X: N data points with m dimensions
      ______N_______
     |              |
     |              |
     M x1 x2 ... xn |
     |              |
     |______________|

     y:    alpha:   W:
      __     _      __
     |y1|   | |    |w1|
     |y2|   | |    |..|
     |..|   N |    |wm|
     |yn|   | |    |__|
     |__|   |_|
'''
class DemoSVM():
    def __init__(self, kernel_type='linear', max_iter=10000, tolerance=0.2, C=1.0):
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.C = C
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.kernel = self.kernels[kernel_type]

    def fit(self, X, y):
        # Initialization
        self.m, self.n = X.shape[0], X.shape[1] # X is a M by N matrix
        if y.shape[0] != self.n or y.shape[1] != 1:
            raise ValueError('the size of training matrix and label matrix doesn\'t match.')

        self.X = X
        self.y = y

        self.alphas = np.zeros((self.n, 1))
        self.unbounded_alpha_index = []
        self.E_list = np.zeros((self.n, 1))
        self.b = 0

        counter = 0
        while True:
            counter += 1

            i, j = self.select_alphas()

            x_i, x_j, y_i, y_j = self.X[:, i], self.X[:, j], self.y[i], self.y[j]

            # Calcluate E
            for k in [i, j]:
                self.E_list[k] = self.E(self.X[:, k], self.y[k])

            # Update alphas
            alpha_1_old, alpha_2_old = np.copy(self.alphas[i]), np.copy(self.alphas[j])

            ita = self.calc_ita(x_i, x_j, y_i, y_j)
            if ita == 0:
                continue
            # Calculate alpha 2
            self.alphas[j] = alpha_2_old + y_j*(self.E_list[i]-self.E_list[j])/ita
            # Clip alpha 2
            (L, H) = self.calc_L_H_bound(y_i, y_j, alpha_1_old, alpha_2_old)
            self.alphas[j] = min(self.alphas[j], H)
            self.alphas[j] = max(self.alphas[j], L)

            # Update alpha 1 according to alpha 2
            self.alphas[i] = alpha_1_old + y_i*y_j*(alpha_2_old - self.alphas[j])

            self.maintain_unbounded_alpha_set(i, j)

            # Update b
            self.update_b(x_i, x_j, y_i, y_j, alpha_1_old, alpha_2_old, self.alphas[i], self.alphas[j])

            # Check KKT conditions
            if not self.violates_KKT_conditions(np.arange(self.n)):
                print('KKT conditions satisfied')
                break
            # Check maximum iteration
            if counter > self.max_iter:
                print('Max iter reached')
                break

            print(self.alphas)
            print(self.unbounded_alpha_index)
        return

    def select_alphas(self):
        if len(self.unbounded_alpha_index) != 0:
            i = self.unbounded_alpha_index[random.randint(0, len(self.unbounded_alpha_index)-1)]

        if len(self.unbounded_alpha_index) == 0 or not (self.violates_KKT_conditions(i)):
            # alphas within 0 and C are empty or consistent
            # loop over whole training set until we
            # find one alpha that violates KKT
            for k in range(0, self.n):
                if k in self.unbounded_alpha_index:
                    continue
                if self.violates_KKT_conditions(k):
                    i = k
                    break

        E_i = self.E_list[i]
        if E_i > 0:
            j = np.argmin(self.E_list)
        else:
            j = np.argmax(self.E_list)

        # ensure i and j are different
        while i == j:
            j = random.randint(0, self.n-1)

        return (i, j)

    def violates_KKT_conditions(self, alpha_index):
        if type(alpha_index) != np.ndarray:
            alpha_index = [alpha_index]

        for index in alpha_index:
            value = self.y[index] * self._g(self.X[:, index])
            if self.alphas[index] == 0:
                if value >= 1 - self.tolerance:
                    continue
                else:
                    return True
            elif self.alphas[index] < self.C:
                if value >= 1 - self.tolerance and value <= 1 + self.tolerance:
                    continue
                else:
                    return True
            elif self.alphas[index] == self.C:
                if value <= 1 + self.tolerance:
                    continue
                else:
                    return True
            else:
                raise ValueError('Internal Error: alpha couldn\'t greater than C')
                return True
        return False

    def maintain_unbounded_alpha_set(self, i, j):
        alphas_new = [(self.alphas[i], i), (self.alphas[j], j)]

        for alpha in alphas_new:
            if alpha[0] == 0 or alpha[0] == self.C:
                # Remove alpha from unbounded alpha set
                if alpha[1] in self.unbounded_alpha_index:
                    self.unbounded_alpha_index.remove(alpha[1])
            else:
                # Add alpha to unbounded alpha set
                if not (alpha[1] in self.unbounded_alpha_index):
                    self.unbounded_alpha_index.append(alpha[1])

    # Calculate the lower and higher bound of the new alphas
    def calc_L_H_bound(self, y_i, y_j, alpha_1_old, alpha_2_old):
        if y_i != y_j:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(self.C, self.C + alpha_2_old - alpha_1_old)
        else:
            L = max(0, alpha_2_old + alpha_1_old - self.C)
            H = min(self.C, alpha_2_old + alpha_1_old)
        return (L, H)

    def update_b(self, x_1, x_2, y_1, y_2, alpha_1_old, alpha_2_old, alpha_1_new, alpha_2_new):
        b1_new = -self.E(x_1, y_1) - y_1*self.kernel(x_1,x_1)*(alpha_1_new - alpha_1_old) - y_2*self.kernel(x_2, x_1)*(alpha_2_new-alpha_2_old) + self.b
        b2_new = -self.E(x_2, y_2) - y_1*self.kernel(x_1,x_2)*(alpha_1_new - alpha_1_old) - y_2*self.kernel(x_2, x_2)*(alpha_2_new-alpha_2_old) + self.b

        if (alpha_1_new > 0 and alpha_1_new < self.C) and (alpha_2_new > 0 and alpha_2_new < self.C):
            self.b = b1_new
        else:
            self.b = (b1_new + b2_new) / 2.0
        return

    # functions
    def E(self, x_i, y_i):
        return self._g(x_i) - y_i

    def calc_ita(self, x_1, x_2, y_1, y_2):
        return self.kernel(x_1.T, x_1) + self.kernel(x_2.T, x_2) - 2*self.kernel(x_1.T, x_2)

    def _g(self, x):
        # y = ( \alpha * y)^T \cdot K(X^T \cdot x) + b
        return np.dot((self.alphas*self.y).T, self.kernel(self.X.T, x)) + self.b

    def _f(self, x):
        return np.sign(self._g(x))

    def predict(self, x):
        # Check the dimension of X
        if x.shape[0] != self.m:
            raise ValueError('The size of prediction data doesn\'t not match the training data.')
        else:
            return self._f(x)

    # Kernels
    def kernel_linear(self, X, x):
        return np.dot(X, x)

    def kernel_quadratic(self, X, x):
        return np.dot(X, x) ** 2
