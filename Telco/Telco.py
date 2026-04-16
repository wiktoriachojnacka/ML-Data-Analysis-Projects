import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D


SEED = 1234
PATH = r"./dane_telco.csv"

df = pd.read_csv(
    PATH,
    sep=";",
    decimal=",",
    skipinitialspace=True,
    na_values=["", "NA", "NaN"]
)

print("Shape:", df.shape)
print(df.head(3))

# 1a Convert YES/NO → 1/0


def yes_no_to01(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.strip()
             .str.lower()
             .map({"yes": 1, "no": 0}))


# detect columns with only yes/no (ignore NaN)
yes_no_cols = []
for col in df.columns:
    vals = (df[col].dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique())
    if len(vals) > 0 and set(vals).issubset({"yes", "no"}):
        yes_no_cols.append(col)

if yes_no_cols:
    df[yes_no_cols] = df[yes_no_cols].apply(yes_no_to01)

print("\nYES/NO columns converted:", yes_no_cols)

# 1b Remove highly correlated cols


def drop_highly_correlated(df_in: pd.DataFrame, threshold: float = 0.98):
    num_df = df_in.select_dtypes(include=[np.number]).copy()
    corr = num_df.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper.columns if any(
        upper[col] > threshold)]
    return cols_to_drop, corr


cols_to_drop_corr, corr_abs = drop_highly_correlated(df, threshold=0.98)
if cols_to_drop_corr:
    print("\nDropping highly correlated cols (>0.98):", cols_to_drop_corr)
    df = df.drop(columns=cols_to_drop_corr)

# 1c Remove target variable churn
if "churn" in df.columns:
    df = df.drop(columns=["churn"])
    print("\nRemoved target column: churn")
else:
    print("\nNo churn column found")


# 1d Histograms & boxplots (numeric)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumber of numeric cols:", len(num_cols))
for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(ax=axes[0], data=df, x=col, kde=False)
    sns.boxplot(ax=axes[1], data=df, y=col, width=0.5)
    axes[0].set_title(f"Histogram: {col}")
    axes[1].set_title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()

# 1e Encode categorical vars
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# fill missing categories
if cat_cols:
    df[cat_cols] = df[cat_cols].fillna("missing")

# fill numeric NaN with median
for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

encoder = OneHotEncoder(handle_unknown="ignore",
                        sparse_output=False).set_output(transform="pandas")

if cat_cols:
    X_cat = encoder.fit_transform(df[cat_cols])
    X_num = df[num_cols].reset_index(drop=True)
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
else:
    X = df[num_cols].copy()

print("\nFeature matrix shape:", X.shape)


# 1f Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 1g Compare distributions
def compare_distributions(raw_df: pd.DataFrame, scaled_arr: np.ndarray, columns, n=6):
    columns = columns[:n]
    scaled_df = pd.DataFrame(scaled_arr, columns=raw_df.columns)

    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(raw_df[col], ax=axes[0], kde=False)
        sns.histplot(scaled_df[col], ax=axes[1], kde=False)
        axes[0].set_title(f"Before scaling: {col}")
        axes[1].set_title(f"After scaling: {col}")
        plt.tight_layout()
        plt.show()


compare_distributions(X, X_scaled, X.columns.tolist(), n=6)


# 1h 3D plots
def plot_3d_if_exists(df_in: pd.DataFrame, col_x: str, col_y: str, col_z: str):
    if all(col in df_in.columns for col in [col_x, col_y, col_z]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = np.array(df_in[col_x])
        y = np.array(df_in[col_y])
        z = np.array(df_in[col_z])
        ax.scatter(x, y, z, marker="o")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_zlabel(col_z)
        plt.title(f"3D: {col_x}, {col_y}, {col_z}")
        plt.show()
    else:
        print(f"(3D) Skipping missing cols: {col_x}, {col_y}, {col_z}")


# use original df (before OHE)
plot_3d_if_exists(df, "longmon", "equipmon", "tollmon")
plot_3d_if_exists(df, "longmon", "equipmon", "wiremon")
plot_3d_if_exists(df, "tollmon", "equipmon", "wiremon")
plot_3d_if_exists(df, "tollmon", "cardmon", "wiremon")


# 2 KMeans: elbow + silhouette
def elbow_and_silhouette(X_arr, k_min=2, k_max=12):
    inertias = []
    sil_scores = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_arr)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_arr, labels))

    # elbow plot
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow method")
    plt.show()

    # silhouette plot
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, sil_scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette vs k")
    plt.show()

    best_k = k_values[int(np.argmax(sil_scores))]
    print("Best k (silhouette):", best_k)
    return k_values, inertias, sil_scores, best_k


k_values, inertias, sil_scores, best_k = elbow_and_silhouette(X_scaled, 2, 12)

# KMeans model
kmeans = KMeans(n_clusters=best_k, random_state=SEED,
                n_init=10).set_output(transform="pandas")
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# 3 Hierarchical clustering (4 clusters)
Z = linkage(X_scaled, method="ward")
labels_hierarchical = fcluster(Z, t=4, criterion="maxclust")

# 4 Evaluation


def visualize_kmeans_clusters(model: KMeans, feature_names):
    for i in range(model.n_clusters):
        plt.figure(figsize=(15, 3))
        pd.Series(model.cluster_centers_[i], index=feature_names)\
            .sort_values().plot(kind="bar")
        plt.title(f"Cluster {i} (standardized)")
        plt.tight_layout()
        plt.show()


def distances_to_centroids(model: KMeans, X_arr: np.ndarray):
    return cdist(X_arr, model.cluster_centers_, metric="euclidean")


def clustering_report(name: str, labels: np.ndarray, X_arr: np.ndarray):
    print(f"\n== {name} ====")
    print("Cluster sizes:")
    print(pd.Series(labels).value_counts().sort_index())

    score = silhouette_score(X_arr, labels)
    print("Silhouette score:", round(score, 4))

    samples = silhouette_samples(X_arr, labels)
    desc = pd.Series(samples).describe()
    print("Silhouette samples summary:")
    print(desc)

    pd.Series(samples).hist(bins=30)
    plt.title(f"Silhouette histogram – {name}")
    plt.xlabel("silhouette")
    plt.ylabel("count")
    plt.show()


# 4a cluster membership
df_membership = pd.DataFrame({
    "cluster_kmeans": labels_kmeans,
    "cluster_hierarchical_4": labels_hierarchical
})
print("\nMembership (first 10):")
print(df_membership.head(10))

# 4b metrics (KMeans)
print("\nKMeans metrics")
print("Inertia:", round(kmeans.inertia_, 4))
print("Score:", round(kmeans.score(X_scaled), 4))

dist_matrix = distances_to_centroids(kmeans, X_scaled)
print("Distance matrix:", dist_matrix.shape)
print("Example (first 5 rows):")
print(pd.DataFrame(dist_matrix[:5]))

print("\nCluster centers (scaled):", kmeans.cluster_centers_.shape)

# 4c centroid visualization
visualize_kmeans_clusters(kmeans, X.columns)

# 4d silhouette comparison
clustering_report("KMeans", labels_kmeans, X_scaled)
clustering_report("Hierarchical (4 clusters)", labels_hierarchical, X_scaled)

print("\nConclusion: interpret clusters via centroids (0 ≈ avg, >0 above, <0 below).")
