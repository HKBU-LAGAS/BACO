import time
import numpy as np
from HashMethod.utils import map_to_consecutive_integers, build_block
from HashMethod.core_spectral_label import spectral_label_core
def spectral_label_Test_hash(
    biadjacency,
    random_state=2025,
    resolution=1.0,
    n_iter=5,
    weight_scheme={'user': 'degree', 'item': 'degree'}
):

    time_begin = time.time()
    rng = np.random.default_rng(random_state)
    num_user, num_item = biadjacency.shape
    n_node = num_user + num_item
    max_cluster = num_user + num_item
    labels = np.arange(max_cluster, dtype=np.int64)
    prob_labels = np.zeros(max_cluster, dtype=np.float64)
    adj = build_block(biadjacency)
    user_deg = np.asarray(biadjacency.sum(axis=1)).squeeze()
    item_deg = np.asarray(biadjacency.sum(axis=0)).squeeze()
    if weight_scheme['user'] == 'CPM': # 1
        user_value = np.ones(num_user, dtype=np.float64)
    elif weight_scheme['user'] == 'degree': # 2
        user_value = user_deg
    elif weight_scheme['user'] == 'sqrt': # 3
        user_value = np.sqrt(user_deg)
    elif weight_scheme['user'] == 'log': # 4
        user_value = np.log1p(user_deg)
    else:
        raise ValueError(f'weight_scheme is not supported {weight_scheme["user"]}!')
    user_sqrt_sum = np.sqrt(np.sum(user_value))
    user_weight = user_value / user_sqrt_sum

    if weight_scheme['item'] == 'CPM':
        item_value = np.ones(num_item, dtype=np.float64)
    elif weight_scheme['item'] == 'degree':
        item_value = np.asarray(biadjacency.T.sum(axis=1)).squeeze()
    elif weight_scheme['item'] == 'sqrt':
        item_value = np.sqrt(item_deg)
    elif weight_scheme['item'] == 'log':
        item_value = np.log1p(item_deg)
    else:
        raise ValueError(f'weight_scheme is not supported {weight_scheme["item"]}!')
    item_sqrt_sum = np.sqrt(np.sum(item_value))
    item_weight = item_deg / item_sqrt_sum

    deg = np.concatenate([user_weight, item_weight])
    sum_edge = 1
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    data = adj.data.astype(np.float64)
    deg = deg.astype(np.float64)
    cluster_sum_U = deg.copy()
    cluster_sum_I = deg.copy()

    for t in range(n_iter):
        nodes = rng.permutation(n_node)
        spectral_label_core(
            num_user, num_item,
            nodes, indptr, indices, data, deg,
            sum_edge, resolution, prob_labels, labels, cluster_sum_U, cluster_sum_I, 0
        )
        # print('Epoch labels num:', len(set(labels)))
    time_cost = time.time() - time_begin

    user_clusters_co = labels[:num_user]
    item_clusters_co = labels[num_user:]
    user_clusters, num_user_clusters = map_to_consecutive_integers(user_clusters_co)
    item_clusters, num_item_clusters = map_to_consecutive_integers(item_clusters_co)

    return time_cost, user_clusters, item_clusters, num_user_clusters, num_item_clusters, user_clusters_co, item_clusters_co