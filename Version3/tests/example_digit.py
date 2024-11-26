import numpy as np
import sys

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

sys.path.append('Version3/')

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.clustering import FailSafeClustering
from tdamapper.plot import MapperPlot


X, y = load_digits(return_X_y=True)
pca_model = PCA(5)
lens = pca_model.fit_transform(X)
pca_model.fit(X) 
total_variance = sum(pca_model.explained_variance_ratio_)  # 總變異

print(total_variance)

# silhouette_for_intervals = []

# for i in range(2, 10):
#     mapper_algo = MapperAlgorithm(
#         cover=CubicalCover(
#             n_intervals=i,
#             overlap_frac=0.35
#         ),
#         clustering=AgglomerativeClustering(linkage='single')
#     )

#     mapper_info = mapper_algo.fit_transform(X, lens)
#     silhouette_for_intervals.append(mapper_info[1])

# print(silhouette_for_intervals)
# best_interval= np.argmax(silhouette_for_intervals) + 2
# print(best_interval)

# document 10, 0.65
# 最佳參數 4, 0.3
mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=4,
        overlap_frac=0.3
    ),
    # clustering=AgglomerativeClustering()
    clustering=FailSafeClustering(clustering=AgglomerativeClustering())
)
mapper_info = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(
    mapper_info[0],
    dim=3,
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
