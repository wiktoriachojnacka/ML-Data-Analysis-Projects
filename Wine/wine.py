import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, confusion_matrix,
    silhouette_score, silhouette_samples
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load data
df = pd.read_csv(
    './data/white_wine.csv',
    sep=';',
    decimal=',',
    skipinitialspace=True,
    na_values=['', 'NA', 'NaN', '?']
)

# ************ EDA ************
print(df.info())
print(df.isna().sum())
print(df.describe())
print("duplicates:", df.duplicated().sum())
print(df[df.duplicated()])

df.hist(figsize=(12, 12))
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

print("EDA conclusion: No missing values; 645 duplicates exist but are acceptable for this "
      "type of data (different wines can share identical chemical measurements). Strongest "
      "positive correlation with quality: alcohol; followed by pH, sulphates, freesulfurdioxide. "
      "Strongest negative correlation: density, chlorides, volatileacidity. Quality (target) "
      "is in the range 1-7 with the most common value being 4. Max values: fixedacidity 11.8, "
      "volatileacidity 1.1, residualsugar 65.8, pH 3.82, alcohol 14.05.")

# ************ DATA PREPARATION ************
seed = 311197

attributes = [
    "fixedacidity", "volatileacidity", "citricacid", "residualsugar",
    "chlorides", "freesulfurdioxide", "totalsulfurdioxide", "density",
    "pH", "sulphates", "alcohol"
]

X = df[attributes]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=seed,
    stratify=y
)

df_train = X_train.copy()
df_train["quality"] = y_train
print(df_train.describe())

#
# ************ CLASSIFICATION MODEL RANDOM FOREST ************
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=seed,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_test_pred)
tolerance_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 1)
mae = mean_absolute_error(y_test, y_test_pred)

print("Test set evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy with +/- 1 tolerance: {tolerance_accuracy:.4f}")
print(f"MAE: {mae:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:")
print(cm)

# ************ CLUSTERING MODEL KMEANS ************
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = range(2, 13)
inertias = []
sils = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_train_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_train_scaled, labels))

# Elbow plot
plt.figure(figsize=(7, 4))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow method (KMeans - train)")
plt.tight_layout()
plt.show()

# Silhouette plot
plt.figure(figsize=(7, 4))
plt.plot(list(ks), sils, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs k (train)")
plt.tight_layout()
plt.show()

best_k = ks[np.argmax(sils)]
print("Best k by silhouette (train set):", best_k)

# Fit final KMeans
kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

df_train_clustered = X_train.copy()
df_train_clustered["cluster"] = train_clusters
df_train_clustered["quality"] = y_train.values

df_test_clustered = X_test.copy()
df_test_clustered["cluster"] = test_clusters
df_test_clustered["quality"] = y_test.values

# Quality per cluster
print("\nMean quality per cluster (train):")
print(df_train_clustered.groupby("cluster")["quality"].mean())
print("\nQuality distribution per cluster (train):")
print(pd.crosstab(df_train_clustered["cluster"],
      df_train_clustered["quality"], normalize="index"))

print("\nMean quality per cluster (test):")
print(df_test_clustered.groupby("cluster")["quality"].mean())
print("\nQuality distribution per cluster (test):")
print(pd.crosstab(df_test_clustered["cluster"],
      df_test_clustered["quality"], normalize="index"))

# Centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X_train.columns)

for i in range(best_k):
    print(f"\nCentroid of cluster {i} (standardized features):")
    print(centroids.loc[i].sort_values(ascending=False))

# Silhouette samples histogram
sil_samples = silhouette_samples(X_train_scaled, train_clusters)
plt.figure(figsize=(7, 4))
plt.hist(sil_samples, bins=30)
plt.title("Silhouette samples (KMeans - train)")
plt.xlabel("silhouette")
plt.ylabel("count")
plt.tight_layout()
plt.show()

print("Silhouette score (train):", round(
    silhouette_score(X_train_scaled, train_clusters), 3))
print("Inertia (train):", round(kmeans.inertia_, 3))

print("Conclusions: Random Forest predicts wine quality reasonably well (~66% exact accuracy), "
      "and the +/- 1 tolerance metric shows ~97% of predictions are off by at most one quality "
      "level. MAE is ~0.4. KMeans grouped wines into two clusters, one with lower quality and "
      "one with higher quality, where the differences are mainly driven by alcohol content, "
      "density, and sugar. The elbow plot shows typical decreasing SSE with no sharp bend in "
      "the middle. The silhouette plot follows a similar pattern. Silhouette samples peak in "
      "the 0.2-0.3 range with the highest count around 120-140 samples.")
