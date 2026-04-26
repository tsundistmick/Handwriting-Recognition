import numpy as np
import matplotlib.pyplot as plt

train_images = np.load('data/processed/train_images.npy')  # (60000, 28, 28)
train_labels = np.load('data/processed/train_labels.npy')  # (60000,)
test_images = np.load('data/processed/test_images.npy')
test_labels = np.load('data/processed/test_labels.npy')

# ===== Визуализация 15 слчайных цифр =====
n = 15
total = len(train_images)
random_indices = np.random.choice(total, n, replace=False)
fig, axes = plt.subplots(3, 5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    idx = random_indices[i]
    ax.imshow(train_images[idx], cmap='gray')
    ax.set_title(train_labels[idx])
    ax.axis('off')

plt.tight_layout()
plt.show()

# ====== Гистограмма распределения для train =====
unique, counts = np.unique(train_labels, return_counts=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(unique, counts, color='skyblue')
plt.xticks(unique)
plt.xlabel('Цифра')
plt.ylabel('Количество изображений')
plt.title('Распределение классов в обучающей выборке MNIST')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50, f'{int(height)}', ha='center', va='bottom')

plt.show()

# ====== Гистограмма распределения для test =====
unique, counts = np.unique(test_labels, return_counts=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(unique, counts, color='skyblue')
plt.xticks(unique)
plt.xlabel('Цифра')
plt.ylabel('Количество изображений')
plt.title('Распределение классов в обучающей выборке MNIST')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50, f'{int(height)}', ha='center', va='bottom')

plt.show()
