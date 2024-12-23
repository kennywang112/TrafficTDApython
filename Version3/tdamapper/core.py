import logging
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from collections import Counter
import numpy as np

from tdamapper.utils.unionfind import UnionFind
from tdamapper._common import ParamsMixin, EstimatorMixin, clone

ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)

def CHS(X, max_clusters=5):

    CH_scores = []

    for k in range(2, max_clusters + 1):

        clustering = AgglomerativeClustering(
            n_clusters=k,
            linkage='ward'
        )
        local_lbls = clustering.fit(X).labels_
        # labels = clustering.fit_predict(X)
        
        # 確保至少有兩個群
        if len(np.unique(local_lbls)) < 2:
            continue
        
        score = calinski_harabasz_score(X, local_lbls)
        CH_scores.append(score)

    if not CH_scores:  # 如果沒有有效的分群結果
        print("Unable to calculate CH scores, defaulting to 1 cluster.")
        return 1, []

    best_cluster = np.argmax(CH_scores) + 2
    print(f'data: {len(X)}, Best cluster N: {best_cluster}')
    return best_cluster, CH_scores

def mapper_labels(X, y, cover, clustering, n_jobs=5):

    def _run_clustering(local_ids):

        clust = clone(clustering)
        local_X = [X[j] for j in local_ids]

        if len(local_X) < 3:

            return local_ids, [-1] * len(local_X), np.nan

        # best_k, CH_in_each_cover = CHS(local_X)

        # if best_k < 2:  # 如果無法形成至少兩個群
        #     return local_ids, [-1] * len(local_X), np.nan

        # clust.set_params(n_clusters=best_k)
        # clust.set_params(n_clusters=2)

        # local_lbls = clustering.fit_predict(X)
        local_lbls = clust.fit(local_X).labels_
        score = calinski_harabasz_score(local_X, local_lbls)
        
        return local_ids, local_lbls, score

    cover_result = list(cover.apply(y))
    _lbls = Parallel(n_jobs)(
        # delayed(_run_clustering)(local_ids) for local_ids in cover.apply(y)
        delayed(_run_clustering)(local_ids) 
        for local_ids in tqdm(cover_result, desc="Processing Clusters")
    )
    itm_lbls = [[] for _ in X]
    CH_values = []
    max_lbl = 0

    for local_ids, local_lbls, CH in _lbls:
        CH_values.append(CH)
        max_local_lbl = 0
        for local_id, local_lbl in zip(local_ids, local_lbls):
            if local_lbl >= 0:
                itm_lbls[local_id].append(max_lbl + local_lbl)
            if local_lbl > max_local_lbl:
                max_local_lbl = local_lbl
        max_lbl += max_local_lbl + 1

    valid_CH = [s for s in CH_values if not np.isnan(s)]  # 過濾掉 NaN 值
    avg_CH = np.mean(valid_CH) if valid_CH else np.nan  # 平均值

    return itm_lbls, avg_CH

def mapper_connected_components(X, y, cover, clustering, n_jobs=5):

    itm_lbls, avg_CH = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
    label_values = set()
    for lbls in itm_lbls:
        label_values.update(lbls)
    uf = UnionFind(label_values)
    for lbls in itm_lbls:
        if len(lbls) > 1:
            for first, second in zip(lbls, lbls[1:]):
                uf.union(first, second)
    labels = [-1 for _ in X]
    for i, lbls in enumerate(itm_lbls):
        # assign -1 to noise points
        root = uf.find(lbls[0]) if lbls else -1
        labels[i] = root
    return labels, avg_CH


def mapper_graph(X, y, cover, clustering, n_jobs=5):

    itm_lbls, avg_CH = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
    graph = nx.Graph()
    for n, lbls in enumerate(itm_lbls):
        for lbl in lbls:
            if not graph.has_node(lbl):
                graph.add_node(lbl, **{ATTR_SIZE: 0, ATTR_IDS: []})
            nodes = graph.nodes()
            nodes[lbl][ATTR_SIZE] += 1
            nodes[lbl][ATTR_IDS].append(n)
    for lbls in itm_lbls:
        lbls_len = len(lbls)
        for i in range(lbls_len):
            source_lbl = lbls[i]
            for j in range(i + 1, lbls_len):
                target_lbl = lbls[j]
                if target_lbl not in graph[source_lbl]:
                    graph.add_edge(source_lbl, target_lbl)
    return graph, avg_CH


def aggregate_graph(X, graph, agg):

    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [X[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class Cover(ParamsMixin):

    def apply(self, X):

        yield list(range(0, len(X)))


class Proximity(Cover):

    def fit(self, X):
        self.__X = X
        return self

    def search(self, x):
        return list(range(0, len(self.__X)))

    def apply(self, X):
        covered_ids = set()
        self.fit(X)
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self.search(xi)
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield neigh_ids


class TrivialCover(Cover):

    def apply(self, X):
        yield list(range(0, len(X)))


class MapperAlgorithm(EstimatorMixin, ParamsMixin):

    def __init__(
        self,
        cover=None,
        clustering=None,
        failsafe=True,
        verbose=True,
        n_jobs=5,
    ):
        self.cover = cover
        self.clustering = clustering
        self.failsafe = failsafe
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y=None):

        X, y = self._validate_X_y(X, y)
        self.__cover = TrivialCover() if self.cover is None \
            else self.cover
        self.__clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        self.__verbose = self.verbose
        self.__failsafe = self.failsafe
        if self.__failsafe:
            self.__clustering = FailSafeClustering(
                clustering=self.__clustering,
                verbose=self.__verbose,
            )
        self.__cover = clone(self.__cover)
        self.__clustering = clone(self.__clustering)
        self.__n_jobs = self.n_jobs
        y = X if y is None else y
        self.graph_= mapper_graph(
            X,
            y,
            self.__cover,
            self.__clustering,
            n_jobs=self.__n_jobs,
        )
        self._set_n_features_in(X)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.graph_


class FailSafeClustering(ParamsMixin):

    def __init__(self, clustering=None, verbose=True):
        self.clustering = clustering
        self.verbose = verbose

    def fit(self, X, y=None):
        self.__clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        self.__verbose = self.verbose
        self.labels_ = None
        try:
            self.__clustering.fit(X, y)
            self.labels_ = self.__clustering.labels_
        except ValueError as err:
            if self.__verbose:
                _logger.warning('Unable to perform clustering on local chart: %s', err)
            self.labels_ = [0 for _ in X]
        return self


class TrivialClustering(ParamsMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.labels_ = [0 for _ in X]
        return self
