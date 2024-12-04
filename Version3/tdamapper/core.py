import logging
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import Counter
import numpy as np

from tdamapper.utils.unionfind import UnionFind
from tdamapper._common import ParamsMixin, EstimatorMixin, clone

print('New version')

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

def elbow_method(X, max_clusters=5):

    CH_scores = []
    # 遍歷群數從 2 到 max_clusters
    for k in range(2, max_clusters + 1):

        clustering = FailSafeClustering(
            AgglomerativeClustering(
                n_clusters=k, 
                linkage='ward'
                ))
        labels = clustering.fit(X).labels_

        # 只有當聚類數量大於 1 且每個聚類至少有 2 個樣本時才計算輪廓係數
        if len(set(labels)) > 1 and all(np.bincount(labels) > 1):
            score = calinski_harabasz_score(X, labels)
        else:
            score = -1

        CH_scores.append(score)
    
    # 找到最高輪廓係數的群數
    best_cluster = np.argmax(silhouette_scores) + 2
    
    return best_cluster

def mapper_labels(X, y, cover, clustering, n_jobs=5):

    # 階層方法
    def _run_clustering(local_ids):

        clust = clone(clustering)
        local_X = [X[j] for j in local_ids]

        if len(local_X) < 3:
            # print(f"Skipping clustering: Too few points ({len(local_X)})")
            return local_ids, [-1] * len(local_X), -1
            # return local_ids, [-1] * len(local_X), 0

        # best_k = elbow_method(local_X)
        best_k = elbow_method(local_X)
        
        clust.set_params(n_clusters=best_k)
        local_lbls = clust.fit(local_X).labels_
        score = silhouette_score(local_X, local_lbls)
        print(f"Best k: {best_k}, Silhouette Score: {score:.3f}")
        
        return local_ids, local_lbls, score

    # 原始模型
    def _run_clustering(local_ids):
        clust = clone(clustering)
        local_X = [X[j] for j in local_ids]
        local_lbls = clust.fit([X[j] for j in local_ids]).labels_

        # Calculate Silhouette Score
        if len(set(local_lbls)) > 1 and all(np.bincount(local_lbls) > 1):  # Silhouette score requires at least 2 clusters
            score = silhouette_score(local_X, local_lbls)
        else:
            # score = -1  # Default score when there is only one cluster or invalid clustering
            score = np.nan

        return local_ids, local_lbls, score
        

    cover_result = list(cover.apply(y))
    _lbls = Parallel(n_jobs)(
        # delayed(_run_clustering)(local_ids) for local_ids in cover.apply(y)
        delayed(_run_clustering)(local_ids) 
        for local_ids in tqdm(cover_result, desc="Processing Clusters")
    )
    itm_lbls = [[] for _ in X]
    silhouette_values = []
    max_lbl = 0

    for local_ids, local_lbls, silhouette in _lbls:
        silhouette_values.append(silhouette)
        max_local_lbl = 0
        for local_id, local_lbl in zip(local_ids, local_lbls):
            if local_lbl >= 0:
                itm_lbls[local_id].append(max_lbl + local_lbl)
            if local_lbl > max_local_lbl:
                max_local_lbl = local_lbl
        max_lbl += max_local_lbl + 1

    # mean silhouette
    # avg_silhouette = np.mean(silhouette_values)
    valid_silhouettes = [s for s in silhouette_values if not np.isnan(s)]  # 過濾掉 NaN 值
    avg_silhouette = np.mean(valid_silhouettes) if valid_silhouettes else np.nan  # 平均值

    # print(f"Average silhouette: {avg_silhouette}")

    return itm_lbls, avg_silhouette

def mapper_connected_components(X, y, cover, clustering, n_jobs=5):
    """
    Identify the connected components of the Mapper graph.

    A connected component is a maximal set of nodes that are reachable from
    each other by following the edges. This function assigns a unique integer
    label to each point in the dataset, based on the connected component of
    the Mapper graph that it belongs to.

    This function uses a union-find data structure to efficiently keep track of
    the connected components as it scans the points of the dataset. This
    approach should be faster than computing the Mapper graph by first calling
    :func:`tdamapper.core.mapper_graph` and then calling
    :func:`networkx.connected_components` on it.

    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: Lens values for the n points of the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    :return: A list of labels. The label at position i identifies the connected
        component of the point at position i in the dataset.
    :rtype: list[int]
    """
    itm_lbls, avg_silhouette = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
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
    return labels, avg_silhouette


def mapper_graph(X, y, cover, clustering, n_jobs=5):
    """
    Create the Mapper graph.

    This function first identifies the unique cluster labels that each point of
    the dataset belongs to. These labels are used to identify the nodes of the
    Mapper graph. Then the edges of the Mapper graph are created by checking if
    any two nodes share some points in their corresponding clusters.

    This function return the Mapper graph as an object of type
    :class:`networkx.Graph`. Each node has an attribute 'size' that stores the
    number of points contained in its corresponding cluster, and an attribute
    'ids' that stores the indices of the points in the dataset that are
    contained in the cluster.

    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: Lens values for the n points of the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    :return: The Mapper graph.
    :rtype: :class:`networkx.Graph`
    """
    itm_lbls, avg_silhouette = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
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
    return graph, avg_silhouette


def aggregate_graph(X, graph, agg):
    """
    Apply an aggregation function to the nodes of a graph.

    This function takes a dataset and a graph, and computes an aggregation
    value for each node of the graph, based on the data points that are
    associated with that node. The aggregation function can be any callable
    that takes a list of values and returns a single value, such as `sum`,
    `mean`, `max`, `min`, etc.

    The function returns a dictionary that maps each node of the graph to its
    aggregation value. The keys of the dictionary are the nodes of the graph,
    and the values are the aggregation values.

    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :param graph: The graph to apply the aggregation function to.
    :type graph: :class:`networkx.Graph`.
    :param agg: The aggregation function to use.
    :type agg: Callable.
    :return: A dictionary of node-aggregation pairs.
    :rtype: dict
    """
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [X[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class Cover(ParamsMixin):
    """
    Abstract interface for cover algorithms.

    This is a naive implementation. Subclasses should override the methods of
    this class to implement more meaningful cover algorithms.
    """

    def apply(self, X):
        """
        Covers the dataset with a single open set.

        This is a naive implementation that returns a generator producing a
        single list containing all the ids if the original dataset. This
        method should be overridden by subclasses to implement more meaningful
        cover algorithms.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        yield list(range(0, len(X)))


class Proximity(Cover):
    """
    Abstract interface for proximity functions. A proximity function is a
    function that maps each point into a subset of the dataset that contains
    the point itself.  Every proximity function defines also a covering
    algorithm based on proximity-netm that is implemented in this class.

    Proximity functions, implemented as subclasses of this class, are a
    convenient way to implement open cover algorithms by using the
    proximity-net construction. Proximity-net is implemented by function
    :func:`tdamapper.core.Proximity.apply`.

    Subclasses should override the methods :func:`tdamapper.core.Proximity.fit`
    and :func:`tdamapper.core.Proximity.search` of this class to implement
    more meaningful proximity functions.
    """

    def fit(self, X):
        """
        Train internal parameters.

        This is a naive implementation that should be overridden by subclasses
        to implement more meaningful proximity functions.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__X = X
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This is a naive implementation that returns all the points in the
        dataset as neighbors. This method should be overridden by subclasses
        to implement more meaningful proximity functions.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: A list containing all the indices of the points in the
            dataset.
        :rtype: list[int]
        """
        return list(range(0, len(self.__X)))

    def apply(self, X):
        """
        Covers the dataset using proximity-net.

        This function applies an iterative algorithm to create the
        proximity-net. It picks an arbitrary point and forms an open cover
        calling the proximity function on the chosen point. The points
        contained in the open cover are then marked as covered, and discarded
        in the following steps. The procedure is repeated on the leftover
        points until every point is eventually covered.

        This function returns a generator that yields each element of the
        proximity-net as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        covered_ids = set()
        self.fit(X)
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self.search(xi)
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield neigh_ids


class TrivialCover(Cover):
    """
    Cover algorithm that covers data with a single subset containing the whole
    dataset.

    This class creates a single open set that contains all the points in the
    dataset.
    """

    def apply(self, X):
        """
        Covers the dataset with a single open set.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        yield list(range(0, len(X)))


class MapperAlgorithm(EstimatorMixin, ParamsMixin):
    """
    A class for creating and analyzing Mapper graphs.

    This class provides two methods :func:`fit` and :func:`fit_transform`. Once
    fitted, the Mapper graph is stored in the attribute `graph_` as a
    :class:`networkx.Graph` object.

    This class adopts the same interface as scikit-learn's estimators for ease
    and consistency of use. However, it's important to note that this is not a
    proper scikit-learn estimator as it does not validata the input in the same
    way as a scikit-learn estimator is required to do. This class can work
    with datasets whose elements are arbitrary objects when feasible for the
    supplied parameters.

    :param cover: A cover algorithm. If no cover is specified,
        :class:`tdamapper.core.TrivialCover` is used, which produces a single
        open cover containing the whole dataset. Defaults to None.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset. If no clustering is specified,
        :class:`tdamapper.core.TrivialClustering` is used, which produces a
        single cluster for each subset. Defaults to None.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param failsafe: A flag that is used to prevent failures. If True, the
        clustering object is wrapped by
        :class:`tdamapper.core.FailSafeClustering`. Defaults to True.
    :type failsafe: bool, optional
    :param verbose: A flag that is used for logging, supplied to
        :class:`tdamapper.core.FailSafeClustering`. If True, clustering
        failures are logged. Set to False to suppress these messages. Defaults
        to True.
    :type verbose: bool, optional
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    """

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
        """
        Create the Mapper graph and store it for later use.

        This method stores the result of :func:`tdamapper.core.mapper_graph` in
        the attribute `graph_` and returns a reference to the calling object.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Lens values for the n points of the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The object itself.
        """
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
        """
        Create the Mapper graph.

        This method is equivalent to calling
        :func:`tdamapper.core.mapper_graph`.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Lens values for the n points of the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The Mapper graph.
        :rtype: :class:`networkx.Graph`
        """
        self.fit(X, y)
        return self.graph_


class FailSafeClustering(ParamsMixin):
    """
    A delegating clustering algorithm that prevents failure.

    This class wraps a clustering algorithm and handles any exceptions that may
    occur during the fitting process. If the clustering algorithm fails,
    instead of throwing an exception, a single cluster containing all points is
    returned. This can be useful for robustness and debugging purposes.

    :param clustering: A clustering algorithm to delegate to.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param verbose: A flag to log clustering exceptions. Set to True to
        enable logging, or False to suppress it. Defaults to True.
    :type verbose: bool, optional.
    """

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
    """
    A clustering algorithm that returns a single cluster.

    This class implements a trivial clustering algorithm that assigns all data
    points to the same cluster. It can be used as an argument of the class
    :class:`tdamapper.core.MapperAlgorithm` to skip clustering in the
    construction of the Mapper graph.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in X]
        return self
