import numpy as np
import pandas as pd

class SoftmaxRegression:
    def __init__(self, learning_rate=0.2, tolerance=0.00001, max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _softmax(self, predictions):
        exp = np.exp(predictions)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape # типа X.shape = (60000, 784) n_samples - образцы/ строки, n_features - признаки/ столбцы
        # cоздаем таблицу, где у каждого образца 1 в столбце класса, которому он принадлежит, в остальных 0
        one_hot_y = pd.get_dummies(y).to_numpy() 

        self.bias = np.zeros(n_classes)
        self.weights = np.zeros((n_features, n_classes)) # матрица признаки x классы
        previous_db = np.zeros(n_classes)
        previous_dw = np.zeros((n_features, n_classes)) 

        for _ in range(self.max_iter):
            y_pred_linear = X @ self.weights + self.bias
            y_pred_softmax = self._softmax(y_pred_linear)
            # считаем разницу м/у предсказанием и реальностью
            error = y_pred_softmax - one_hot_y
            # считаем градиенты
            db = 1 / n_samples * np.sum(error, axis=0)
            dw = 1 / n_samples * X.T @ error

            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw
            abs_db_reduction = np.abs(db - previous_db)
            abs_dw_reduction = np.abs(dw - previous_dw)

            if np.mean(abs_db_reduction) < self.tolerance and np.mean(abs_dw_reduction) < self.tolerance: # проверяем, что среднее значение весов поменялось не больше чем порог
                break

            previous_db = db
            previous_dw = dw

    def predict(self, X_test):
        y_pred_linear = X_test @ self.weights + self.bias
        y_pred_softmax = self._softmax(y_pred_linear)
        # получаем наиболее вероятные классы
        most_prob_classes = np.argmax(y_pred_softmax, axis=1)

        return most_prob_classes