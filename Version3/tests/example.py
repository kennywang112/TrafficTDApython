import numpy as np
import sys
import pandas as pd

from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path.append('Version3/')

from tdamapper.core import MapperAlgorithm, FailSafeClustering
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

X, y = make_circles(                # load a labelled dataset
    n_samples=5000,
    noise=0.05,
    factor=0.3,
    random_state=42
)

pca = PCA(n_components=2)
lens = pca.fit_transform(X)

explained_variance_ratio  = pca.explained_variance_ratio_
print(explained_variance_ratio)


# silhouette_for_intervals = []

# for i in range(2, 10):
#     mapper_algo = MapperAlgorithm(
#         cover=CubicalCover(
#             n_intervals=i,
#             overlap_frac=0.4
#         ),
#         clustering=AgglomerativeClustering()
#     )

#     mapper_info = mapper_algo.fit_transform(X, lens)
#     silhouette_for_intervals.append(mapper_info[1])

# print(silhouette_for_intervals)
# best_interval= np.argmax(silhouette_for_intervals) + 2 # 索引從 0 開始，迴圈從 2 開始，所以1+1
# print(best_interval)

# # 10, 0.65 原始參數
# mapper_algo = MapperAlgorithm(
#     cover=CubicalCover(
#         n_intervals=5,
#         overlap_frac=0.38
#     ),
#     # clustering=AgglomerativeClustering()
#     clustering=FailSafeClustering(clustering=AgglomerativeClustering())
# )
# mapper_info = mapper_algo.fit_transform(X, lens)

# mapper_plot = MapperPlot(
#     mapper_info[0],
#     dim=2,
#     iterations=60,
#     seed=42
# )

# fig = mapper_plot.plot_plotly(
#     title='',
#     width=600,
#     height=600,
#     colors=y,                       # color according to categorical values
#     cmap='jet',                     # Jet colormap, for classes
#     agg=np.nanmean,                 # aggregate on nodes according to mean
# )

# fig.show(config={'scrollZoom': True})

# mapper_plot.plot_plotly_update(
#     fig,                            # reuse the plot with the same positions
#     colors=y,
#     cmap='viridis',                 # viridis colormap, for ranges
#     agg=np.nanstd,                  # aggregate on nodes according to std
# )

# fig.show(config={'scrollZoom': True})
