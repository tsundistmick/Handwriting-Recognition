import numpy as np
from pathlib import Path
from LogReg.softmax_regression import SoftmaxRegression 
from sklearn.metrics import accuracy_score

data_path = Path("data/processed")

X_train = np.load(data_path / "train_images.npy")
y_train = np.load(data_path / "train_labels.npy")
X_test = np.load(data_path / "test_images.npy")
y_test = np.load(data_path / "test_labels.npy")

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
#меняем из 28x28 в плоские вектора
X_train = X_train.reshape(X_train.shape[0], 784) 
X_test = X_test.reshape(X_test.shape[0], 784)

softmax_regression = SoftmaxRegression()
softmax_regression.fit(X_train, y_train)
softmax_pred_res = softmax_regression.predict(X_test)
softmax_accuracy = accuracy_score(y_test, softmax_pred_res)

print(f'Softmax-regression accuracy: {softmax_accuracy}')