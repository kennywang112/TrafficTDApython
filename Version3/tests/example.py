import numpy as np
import sys
import pandas as pd
import pickle

from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

sys.path.append('Version3/')

from tdamapper.core import MapperAlgorithm, FailSafeClustering
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

import matplotlib.pyplot as plt

X, y = make_circles(                # load a labelled dataset
    n_samples=5000,
    noise=0.03,
    factor=0.3,
    random_state=42
)

pca = PCA(n_components=2)
lens = pca.fit_transform(X)

explained_variance_ratio  = pca.explained_variance_ratio_

detailed_results = []

for overlap in range(3, 6):
    silhouette_for_intervals = []

    for interval in range(5, 11):
        mapper_algo = MapperAlgorithm(
            cover=CubicalCover(
                n_intervals=interval,
                overlap_frac=overlap / 10
            ),
            clustering=AgglomerativeClustering(linkage='single')
        )

        mapper_info = mapper_algo.fit_transform(X, lens)
        silhouette_for_intervals.append(mapper_info[1])

        detailed_results.append({
            "overlap": overlap,
            "interval": interval,
            "silhouette": mapper_info[1],
            "mapper_info": mapper_info
        })
 
    best_interval = np.argmax(silhouette_for_intervals) + 5  # +5 因為 interval 從5開始

# output_file = '/Users/wangqiqian/Desktop/TrafficTDApython/Version3/GridSearch/example_grid.pkl'
# with open(output_file, 'wb') as f:
#     pickle.dump(detailed_results, f)

print('sorted data')

detailed_results_df = pd.DataFrame(detailed_results)
sorted = detailed_results_df.sort_values(by='silhouette')
print(sorted[['silhouette', 'interval', 'overlap']])

# 10, 0.65 document參數
mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=sorted['interval'].iloc[-1],
        overlap_frac=sorted['overlap'].iloc[-1]/10
    ),
    clustering=AgglomerativeClustering(linkage='single')
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

# mapper_plot.plot_plotly_update(
#     fig,                            # reuse the plot with the same positions
#     colors=y,
#     cmap='viridis',                 # viridis colormap, for ranges
#     agg=np.nanstd,                  # aggregate on nodes according to std
# )

# fig.show(config={'scrollZoom': True})
