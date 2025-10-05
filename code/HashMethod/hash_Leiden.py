import time
from sknetwork.clustering import Leiden
from spectral_label_kit import build_block
from HashMethod.utils import map_to_consecutive_integers, double_map_to_consecutive_integers
import numpy as np
from scipy import sparse
from sknetwork.utils.check import get_probs
from sknetwork.utils.format import check_format, get_adjacency, directed2undirected

class CustomLouvain(Leiden):
    def _pre_processing(self, input_matrix, force_bipartite):
        """Pre-processing for Louvain.

         Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite :
            If ``True``, force the input matrix to be considered as a biadjacency matrix even if square.

        Returns
        -------
        adjacency :
            Adjacency matrix.
        out_weights, in_weights :
            Node weights.
        membership :
            Membership matrix (labels).
        index :
            Index of nodes.
        """
        self._init_vars()

        # adjacency matrix
        input_matrix = check_format(input_matrix)
        force_directed = self.modularity == 'dugue'
        adjacency, self.bipartite = get_adjacency(input_matrix, force_directed=force_directed,
                                                  force_bipartite=force_bipartite)

        # shuffling
        n = adjacency.shape[0]
        index = np.arange(n)
        if self.shuffle_nodes:
            index = self.random_state.permutation(index)
            adjacency = adjacency[index][:, index]

        # node weights
        if self.modularity == 'potts':
            out_weights = get_probs('uniform', adjacency)
            in_weights = out_weights.copy()
        elif self.modularity == 'newman':
            out_weights = get_probs('degree', adjacency)
            in_weights = out_weights.copy()
        elif self.modularity == 'dugue':
            out_weights = get_probs('degree', adjacency)
            in_weights = get_probs('degree', adjacency.T)
        elif self.modularity == 'cpm':
            user_num, item_num = adjacency.get_shape()
            out_weights = np.ones(user_num)
            in_weights = np.ones(item_num)
        elif self.modularity == 'baco':
            user_num, item_num = adjacency.get_shape()
            out_weights = np.asarray(adjacency.sum(axis=1)).squeeze()
            in_weights = np.ones(item_num)
            out_weights = out_weights / np.sqrt(np.sum(out_weights))
            in_weights = in_weights / np.sqrt(np.sum(in_weights))
        else:
            raise ValueError(f'CustomLeiden: Unknown modularity function {self.modularity}.')

        # normalized, symmetric adjacency matrix (sums to 1)
        adjacency = directed2undirected(adjacency)
        if self.modularity in ['newman', 'dugue','potts']:
            adjacency = adjacency / adjacency.data.sum()

        # cluster membership
        membership = sparse.identity(n, format='csr')

        return adjacency, out_weights, in_weights, membership, index

def Leiden_hash(
    biadjacency,
    num_users_c,
    num_items_c,
    resolution=1.0,
    random_state=2025,
    mod_type='degree',
    mylog=None,
    Secondary_Clusters=False,
):
    # print('input num cluster:',num_users_c, num_items_c)
    adj = build_block(biadjacency)
    num_user, num_item = biadjacency.shape
    user_deg = np.asarray(biadjacency.sum(axis=1)).squeeze()
    item_deg = np.asarray(biadjacency.sum(axis=0)).squeeze()
    user_deg = user_deg / np.sum(user_deg)
    item_deg = item_deg / np.sum(item_deg)
    deg = np.concatenate([user_deg, item_deg])
    adj = adj / adj.data.sum()
    cluster_sum_U = np.zeros_like(deg)
    cluster_sum_I = np.zeros_like(deg)
    n_node = num_user + num_item
    time_begin = time.time()
    leiden = CustomLouvain(resolution=resolution, modularity=mod_type)
    leiden.fit(biadjacency, force_bipartite=True)
    if Secondary_Clusters == False:
        time_cost = time.time() - time_begin
        user_clusters_co = leiden.labels_row_
        item_clusters_co = leiden.labels_col_
        user_clusters, num_user_clusters = map_to_consecutive_integers(user_clusters_co)
        item_clusters, num_item_clusters = map_to_consecutive_integers(item_clusters_co)
    else :
        print('Secondary Clustering')
        labels = np.concatenate([leiden.labels_row_, leiden.labels_col_])
        from collections import defaultdict
        from HashMethod.utils import map_to_consecutive_integers_multi
        multi_label = [[labels[_]] for _ in range(n_node)]
        for x in range(num_user):
            cluster_sum_U[labels[x]] += deg[x]
        for x in range(num_user, n_node):
            cluster_sum_I[labels[x]] += deg[x]
        print('Secondary Clustering on', len(set(labels)), 'clusters')
        user_label = set(labels[:num_user])
        for x in range(num_user):
            prob_labels = defaultdict(float)
            # maintain mini community
            for i in range(adj.indptr[x], adj.indptr[x + 1]):
                y = adj.indices[i]
                l = labels[y]
                if l not in user_label: continue
                prob_labels[l] += adj.data[i]
            max_value = None
            label_now = labels[x]
            label_best = labels[x]
            for label_target, now_value in prob_labels.items():
                now_value = now_value - prob_labels.get(label_now, 0) - resolution * deg[x] * (
                            cluster_sum_I[label_target] - cluster_sum_I[label_now])
                if label_target != label_now:
                    if (max_value is None) or (now_value > max_value):
                        max_value = now_value
                        label_best = label_target
            if label_now == label_best: continue
            top_labels = [label_now, label_best]
            multi_label[x] = top_labels

        time_cost = time.time() - time_begin
        user_clusters_co = multi_label[:num_user]
        item_clusters_co = multi_label[num_user:]
        user_clusters, num_user_clusters, user_count = map_to_consecutive_integers_multi(user_clusters_co)
        item_clusters, num_item_clusters, item_count = map_to_consecutive_integers_multi(item_clusters_co)
        mylog.write('Overlap Ratio user clusters:', user_count / num_user, 'item clusters:', item_count / num_item)
        mylog.write('increase nodes:', user_count - num_user)

    return time_cost, user_clusters, item_clusters, num_user_clusters, num_item_clusters, user_clusters_co, item_clusters_co