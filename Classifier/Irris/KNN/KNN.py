import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data  # Lấy toàn bộ chiều dài, chiều rộng đài hoa và cánh hoa
y = iris.target
X_df = pd.DataFrame(X, columns=iris.feature_names)

correlation_matrix = X_df.corr()
# Kiểm tra giá trị bị thiếu
missing_values = X_df.isnull().sum()
print("Số lượng giá trị bị thiếu trong mỗi đặc trưng:")
print(missing_values)

# In các hàng chứa giá trị bị thiếu
if X_df.isnull().values.any():
    print("\nCác hàng chứa giá trị bị thiếu:")
    print(X_df[X_df.isnull().any(axis=1)])
else:
    print("\nKhông có giá trị bị thiếu trong dữ liệu.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Giảm chiều dữ liệu xuống 2 và 3 chiều bằng PCA
pca_2d = PCA(n_components=2)
X_combined_pca_2d = pca_2d.fit_transform(X_combined_std)
X_train_pca_2d = pca_2d.transform(X_train)
X_test_pca_2d = pca_2d.transform(X_test)

pca_3d = PCA(n_components=3)
X_combined_pca_3d = pca_3d.fit_transform(X_combined_std)
X_train_pca_3d = pca_3d.transform(X_train)
X_test_pca_3d = pca_3d.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_pca_2d, y_train)

def plot_decision_regions_2d(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none',
                    edgecolor='black', alpha=0.3,
                    linewidth=3, marker='o',
                    s=100, label='test set')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_decision_regions_3d(X, y, classifier):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], X[y == cl, 2],
                   c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

y_pred = knn.predict(X_test_pca_2d)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(conf_matrix)

plot_decision_regions_2d(X_combined_pca_2d, y_combined, classifier=knn, test_idx=range(105, 150))

# Sử dụng PCA 3D để vẽ biểu đồ 3D với cùng bộ dữ liệu
knn_3d = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn_3d.fit(X_train_pca_3d, y_train)
plot_decision_regions_3d(X_combined_pca_3d, y_combined, classifier=knn_3d)

plt.figure(figsize=(10,4)) 
sns.heatmap(correlation_matrix, annot=True, cmap='cubehelix_r')
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()