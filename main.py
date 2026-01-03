import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Load images from directory ===
def load_data(data_dir, target_size=(64, 64), batch_size=5000, allowed_classes=None):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    images, labels = next(generator)
    class_indices = generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    if allowed_classes:
        keep_indices = [i for i, label in enumerate(labels) if idx_to_class[int(label)] in allowed_classes]
        images = images[keep_indices]
        labels = labels[keep_indices]
        filtered_classes = sorted(set(idx_to_class[int(label)] for label in labels))
        class_map = {cls: i for i, cls in enumerate(filtered_classes)}
        labels = np.array([class_map[idx_to_class[int(label)]] for label in labels])
    else:
        class_map = class_indices
        labels = labels.astype(int)

    return images.reshape(images.shape[0], -1), labels, list(class_map.keys())

# === Plot confusion matrix ===
def plot_confusion(cm, title, labels, color):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted', labelpad=15)
    plt.ylabel('Actual', labelpad=15)
    plt.tight_layout()
    plt.show()

# === Paths and classes ===
train_dir = './fruits_dataset'
test_dir = './fruits_dataset'
selected_classes = ['Apple', 'banana', 'orange']

from sklearn.model_selection import train_test_split

# === Load full data once ===
X_all, Y_all, label_names = load_data(train_dir, allowed_classes=selected_classes)
print(f"Total images loaded: {X_all.shape[0]}  |  Classes: {label_names}")

# === PCA before split ===
pca = PCA(n_components=0.95, random_state=42)
X_all_pca = pca.fit_transform(X_all)

# === Train-test split ===
X_train_pca, X_test_pca, Y_train, Y_test = train_test_split(
    X_all_pca, Y_all, test_size=0.2, random_state=42, stratify=Y_all
)


# === Naive Bayes ===
print("\n" + "=" * 60)
print(f"{'Naive Bayes Classifier':^60}")
print("=" * 60)

start_time = time.time()
nb_model = GaussianNB()
nb_model.fit(X_train_pca, Y_train)
nb_pred = nb_model.predict(X_test_pca)
end_time = time.time()

print(f"{'Time taken':<20}: {end_time - start_time:.4f} seconds")
print(f"{'Accuracy':<20}: {accuracy_score(Y_test, nb_pred)*100:.2f}%")

# --- Formatted classification report
report = classification_report(Y_test, nb_pred, target_names=label_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].round(2).astype(str)
df['support'] = df['support'].astype(int)

print("\n" + "-" * 70)
print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")
print("-" * 70)
for label in label_names:
    row = df.loc[label]
    print(f"{label:<15}{row['precision']:>12}{row['recall']:>12}{row['f1-score']:>12}{row['support']:>12}")
print("-" * 70)
print(f"{'Accuracy':<15}{'':>36}{accuracy_score(Y_test, nb_pred)*100:>11.2f}%")
print("-" * 70)

plot_confusion(confusion_matrix(Y_test, nb_pred), "Naive Bayes Confusion Matrix", label_names, 'Blues')

# === Decision Tree ===
print("\n" + "=" * 60)
print(f"{'Decision Tree Classifier':^60}")
print("=" * 60)

param_grid = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
     'min_samples_leaf': [1, 2]
 }

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0  
)


start_time = time.time()
grid_dt.fit(X_train_pca, Y_train)
best_dt = grid_dt.best_estimator_
dt_pred = best_dt.predict(X_test_pca)
end_time = time.time()

print(f"{'Time taken':<20}: {end_time - start_time:.4f} seconds")
print(f"{'Accuracy':<20}: {accuracy_score(Y_test, dt_pred)*100:.2f}%")

report = classification_report(Y_test, dt_pred, target_names=label_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].round(2).astype(str)
df['support'] = df['support'].astype(int)

print("\n" + "-" * 70)
print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")
print("-" * 70)
for label in label_names:
    row = df.loc[label]
    print(f"{label:<15}{row['precision']:>12}{row['recall']:>12}{row['f1-score']:>12}{row['support']:>12}")
print("-" * 70)
print(f"{'Accuracy':<15}{'':>36}{accuracy_score(Y_test, dt_pred)*100:>11.2f}%")
print("-" * 70)

plot_confusion(confusion_matrix(Y_test, dt_pred), "Optimized Decision Tree Confusion Matrix", label_names, 'Oranges')

tree_rules = export_text(best_dt, feature_names=[f"PC{i}" for i in range(X_train_pca.shape[1])], max_depth=3)
#print("\nDecision Path Rules (depth ≤ 3):\n", tree_rules)

plt.figure(figsize=(20, 10))
plot_tree(best_dt, max_depth=2, feature_names=[f"PC{i}" for i in range(X_train_pca.shape[1])],
          class_names=label_names, filled=True, rounded=True)
plt.title("Decision Tree Visualization (Top 2 Levels)")
plt.show()

# === MLP ===
print("\n" + "=" * 60)
print(f"{'MLP Neural Network Classifier':^60}")
print("=" * 60)

start_time = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                          solver='adam', max_iter=300, early_stopping=True, random_state=42)
mlp_model.fit(X_train_pca, Y_train)
mlp_pred = mlp_model.predict(X_test_pca)
end_time = time.time()

print(f"{'Time taken':<20}: {end_time - start_time:.4f} seconds")
print(f"{'Accuracy':<20}: {accuracy_score(Y_test, mlp_pred)*100:.2f}%")

report = classification_report(Y_test, mlp_pred, target_names=label_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].round(2).astype(str)
df['support'] = df['support'].astype(int)

print("\n" + "-" * 70)
print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")
print("-" * 70)
for label in label_names:
    row = df.loc[label]
    print(f"{label:<15}{row['precision']:>12}{row['recall']:>12}{row['f1-score']:>12}{row['support']:>12}")
print("-" * 70)
print(f"{'Accuracy':<15}{'':>36}{accuracy_score(Y_test, mlp_pred)*100:>11.2f}%")
print("-" * 70)

plot_confusion(confusion_matrix(Y_test, mlp_pred), "MLP (Neural Network) Confusion Matrix", label_names, 'Greens')

# === Summary ===
print("\n" + "=" * 60)
print(f"{'Model Performance Summary':^60}")
print("=" * 60)
print(f"{'Model':<35}{'Accuracy (%)':>25}")
print("-" * 60)
print(f"{'Naive Bayes':<35}{accuracy_score(Y_test, nb_pred)*100:>25.2f}")
print(f"{'Decision Tree (Tuned)':<35}{accuracy_score(Y_test, dt_pred)*100:>25.2f}")
print(f"{'MLP Neural Network':<35}{accuracy_score(Y_test, mlp_pred)*100:>25.2f}")
print("=" * 60)
