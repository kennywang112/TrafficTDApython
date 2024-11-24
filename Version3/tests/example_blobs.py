import numpy as np
import sys
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path.append('Version3/')

from tdamapper.core import MapperAlgorithm, FailSafeClustering
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

# X, y = make_moons(
#     n_samples=5000,
#     noise=0.05
# )

X, y = make_blobs(n_samples=5000, random_state=170)

# from umap import UMAP
# import matplotlib.pyplot as plt
# # Apply UMAP for dimensionality reduction
# umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs=-1)
# X_umap = umap.fit_transform(X)

# # Plot UMAP results
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', s=5)
# plt.colorbar(scatter, label='Class Label')
# plt.title("UMAP Projection of Moon Dataset")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.show()

lens = PCA(2).fit_transform(X)

# silhouette_for_intervals = []

# for i in range(2, 10):
#     mapper_algo = MapperAlgorithm(
#         cover=CubicalCover(
#             n_intervals=i,
#             overlap_frac=0.3
#         ),
#         clustering=AgglomerativeClustering()
#     )

#     mapper_info = mapper_algo.fit_transform(X, lens)
#     silhouette_for_intervals.append(mapper_info[1])

# print(silhouette_for_intervals)
# best_interval= np.argmax(silhouette_for_intervals) + 2
# print(best_interval)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=2,
        overlap_frac=0.3
    ),
    # clustering=AgglomerativeClustering()
    clustering=FailSafeClustering(clustering=AgglomerativeClustering())
)
mapper_info = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(
    mapper_info[0],
    dim=2,
    iterations=60,
    seed=42
)

fig = mapper_plot.plot_plotly(
    title='',
    width=600,
    height=600,
    colors=y,                       # color according to categorical values
    cmap='jet',                     # Jet colormap, for classes
    agg=np.nanmean,                 # aggregate on nodes according to mean
)

fig.show(config={'scrollZoom': True})
