import numpy as np
import sys

from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering

sys.path.append('Version3/')

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

# X, y = make_circles(                # load a labelled dataset
#     n_samples=5000,
#     noise=0.05,
#     factor=0.3,
#     random_state=42
# )

X, y = make_moons(
    n_samples=5000,
    noise=0.05
)

lens = PCA(2).fit_transform(X)

# sse_for_interval = []

# for i in range(1, 10):
#     mapper_algo = MapperAlgorithm(
#         cover=CubicalCover(
#             n_intervals=i,
#             overlap_frac=0.3
#         ),
#         clustering=AgglomerativeClustering()
#     )

#     mapper_info = mapper_algo.fit_transform(X, lens)

#     print(mapper_info[1])
#     sse_for_interval.append(mapper_info[1])

# print(sse_for_interval)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=6,
        overlap_frac=0.5
    ),
    clustering=AgglomerativeClustering()
)
mapper_info = mapper_algo.fit_transform(X, lens)

print(mapper_info)
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

# mapper_plot.plot_plotly_update(
#     fig,                            # reuse the plot with the same positions
#     colors=y,
#     cmap='viridis',                 # viridis colormap, for ranges
#     agg=np.nanstd,                  # aggregate on nodes according to std
# )

# fig.show(config={'scrollZoom': True})
