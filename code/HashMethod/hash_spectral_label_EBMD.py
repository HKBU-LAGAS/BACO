import time
import numpy as np
from HashMethod.utils import map_to_consecutive_integers, build_block
from HashMethod.core_spectral_label_EBMD import spectral_label_EBMD_core
def spectral_label_EBMD_hash(
    biadjacency,
    num_users_c,
    num_items_c,
    resolution=1.0,
    n_iter=10,
    random_state=2025,
):
    print('input num cluster:',num_users_c, num_items_c, num_users_c + num_items_c)

    time_begin = time.time()
    rng = np.random.default_rng(random_state)
    num_user, num_item = biadjacency.shape
    n_node = num_user + num_item
    max_cluster = num_user + num_item
    labels = np.arange(max_cluster, dtype=np.int64)
    edge_labels = np.zeros(max_cluster, dtype=np.float64)
    edge_weights = np.zeros(max_cluster, dtype=np.float64)
    prob_labels = np.zeros(max_cluster, dtype=np.float64)
    adj = build_block(biadjacency)
    deg = np.asarray(adj.sum(axis=1)).squeeze()
    sum_edge = np.sum(deg)
    label_set = set()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    data = adj.data.astype(np.float64)
    deg = deg.astype(np.float64)
    cluster_sum_U = deg.copy()
    cluster_sum_I = deg.copy()

    for t in range(n_iter):
        nodes = rng.permutation(n_node)
        # Update labels for each node in random order
        # for x in nodes:
        #     label_set.clear()
        #     for i in range(adj.indptr[x], adj.indptr[x + 1]):
        #         y = adj.indices[i]
        #         l = labels[y]
        #         edge_weights[l] += adj.data[i]
        #         prob_labels[l] += deg[y] / sum_edge
        #         label_set.add(l)
        #
        #     max_value = -100000
        #     label_best = labels[x]
        #     ori_edge = edge_weights[labels[x]]
        #     add_edge = 0
        #     for label_target in label_set:
        #         add_u = 0
        #         add_v = 0
        #         if x < num_user:
        #             add_u = 1
        #         else :
        #             add_v = 1
        #         Cu = count_labels_U[label_target] + add_u
        #         Cv = count_labels[label_target] - count_labels_U[label_target] + add_v
        #         if Cu * Cv == 0:
        #             raise ValueError(f'Division by zero in label {label_target} with Cu={Cu}, Cv={Cv}')
        #
        #         gamma = (edge_labels[label_target] + edge_weights[label_target]) / (Cu * Cv) - sum_edge / (num_user * num_item)
        #         prob = gamma * edge_weights[label_target] - resolution * gamma * gamma * deg[x] * prob_labels[label_target]
        #         if prob > max_value:
        #             max_value = prob
        #             label_best = label_target
        #             add_edge = edge_weights[label_target]
        #
        #         edge_weights[label_target] = 0
        #         prob_labels[label_target] = 0
        #     if label_best != labels[x]:
        #         count_labels[labels[x]] -= 1
        #         count_labels[label_best] += 1
        #         edge_labels[labels[x]] -= ori_edge
        #         edge_labels[label_best] += add_edge
        #         if x < num_user:
        #             count_labels_U[labels[x]] -= 1
        #             count_labels_U[label_best] += 1
        #         labels[x] = label_best

        # Uncomment the following line to use the Cython implementation
        spectral_label_EBMD_core(
            num_user, num_item, nodes, indptr, indices, data, deg, sum_edge,
            resolution, prob_labels, edge_weights, edge_labels, labels, cluster_sum_U, cluster_sum_I,
        )

        print('Epoch labels num:', len(set(labels[:num_user])) + len(set(labels[num_user:])), 'target:', num_users_c + num_items_c)
    time_cost = time.time() - time_begin

    user_clusters_co = labels[:num_user]
    item_clusters_co = labels[num_user:]
    user_clusters, num_user_clusters = map_to_consecutive_integers(user_clusters_co)
    item_clusters, num_item_clusters = map_to_consecutive_integers(item_clusters_co)

    return time_cost, user_clusters, item_clusters, num_user_clusters, num_item_clusters, user_clusters_co, item_clusters_co