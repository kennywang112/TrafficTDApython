from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# 生成僅一個聚集中心的數據
X, _ = make_blobs(n_samples=10, centers=1, cluster_std=0.1, random_state=42)

def calculate_sse(X, labels):
    """
    根據聚類結果手動計算 SSE。
    """
    sse = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        center = cluster_points.mean(axis=0)
        sse += np.sum((cluster_points - center) ** 2)
    return sse

SSE = []
k_max = 10
for k in range(1, k_max):  # 從 cluster=1 開始
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(X)
    sse = calculate_sse(X, labels)
    SSE.append(sse)

print("SSE values:", SSE)