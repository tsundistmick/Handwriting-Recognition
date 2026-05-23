import numpy as np
from pathlib import Path
from softmax_regression import SoftmaxRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import save_results_to_json 

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

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

softmax_regression = SoftmaxRegression()
softmax_regression.fit(X_train, y_train)
softmax_pred_res = softmax_regression.predict(X_test)
softmax_accuracy = accuracy_score(y_test, softmax_pred_res)

print(f'Softmax-regression accuracy: {softmax_accuracy}')


sk_softmax_regression = LogisticRegression(max_iter=1000)
sk_softmax_regression.fit(X_train, y_train)
sk_softmax_pred_res = sk_softmax_regression.predict(X_test)
sk_softmax_accuracy = accuracy_score(y_test, sk_softmax_pred_res)

print(f'sk Softmax-regression accuracy: {sk_softmax_accuracy}')

results = {
    "model_comparison": {
        "my_softmax_regression": {
            "accuracy": float(softmax_accuracy),
            "parameters": {
                "learning_rate": softmax_regression.learning_rate,
                "max_iter": softmax_regression.max_iter,
                "tolerance": softmax_regression.tolerance
            }
        },
        "sklearn_logistic_regression": {
            "accuracy": float(sk_softmax_accuracy),
            "parameters": {
                "max_iter": 1000,
                "multi_class": "multinomial",
                "solver": "lbfgs"
            }
        }
    },
    "dataset_info": {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": 10  
    }
}

save_results_to_json(results, "softmax_regression_results.json")

print(f"\nAccuracy difference: {abs(softmax_accuracy - sk_softmax_accuracy):.4f}")
