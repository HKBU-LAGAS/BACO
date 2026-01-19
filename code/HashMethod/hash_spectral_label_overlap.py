import time
import numpy as np
from HashMethod.utils import map_to_consecutive_integers_multi, build_block
from HashMethod.core_spectral_label import spectral_label_core
from HashMethod.core_second_cluster import Second_Cluster_core

def spectral_label_overlap_hash(
    biadjacency,
    num_users_c,
    num_items_c,
    resolution=1.0,
    n_iter=5,
    random_state=2025,
    weight_scheme={'user': 'degree', 'item': 'degree'},
    mylog=None,
    budget=None,
):
    time_begin = time.time()
    rng = np.random.default_rng(random_state)
    num_user, num_item = biadjacency.shape
    n_node = num_user + num_item
    max_cluster = num_user + num_item
    labels = np.arange(max_cluster, dtype=np.int64)
    prob_labels = np.zeros(max_cluster, dtype=np.float32)
    adj = build_block(biadjacency)
    user_deg = np.asarray(biadjacency.sum(axis=1)).squeeze()
    item_deg = np.asarray(biadjacency.sum(axis=0)).squeeze()
    if weight_scheme['user'] == 'CPM':  # 1
        user_value = np.ones(num_user, dtype=np.float32)
    elif weight_scheme['user'] == 'degree':  # 2
        user_value = user_deg
    elif weight_scheme['user'] == 'sqrt':  # 3
        user_value = np.sqrt(user_deg)
    elif weight_scheme['user'] == 'log':  # 4
        user_value = np.log1p(user_deg)
    else:
        raise ValueError(f'weight_scheme is not supported {weight_scheme["user"]}!')
    user_sqrt_sum = np.sqrt(np.sum(user_value))
    user_weight = user_value / user_sqrt_sum

    if weight_scheme['item'] == 'CPM':
        item_value = np.ones(num_item, dtype=np.float32)
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
    data = adj.data.astype(np.float32)
    deg = deg.astype(np.float32)
    # cluster_sum_U = np.concatenate([user_weight, np.zeros(num_item, dtype=np.float32)])
    # cluster_sum_I = np.concatenate([np.zeros(num_user, dtype=np.float32), item_weight])
    cluster_sum_U = deg.copy()
    cluster_sum_I = deg.copy()

    for t in range(n_iter):
        if len(set(labels[:num_user]))+len(set(labels[num_user:])) <= budget : break
        nodes = rng.permutation(n_node)
        spectral_label_core(
            num_user, num_item,
            nodes, indptr, indices, data, deg,
            sum_edge, resolution, prob_labels, labels, cluster_sum_U, cluster_sum_I, 0
        )
        # print('Epoch labels num:', len(set(labels)))

    multi_label = Second_Cluster_core(
        num_user, num_item,
        nodes, indptr, indices, data, deg,
        sum_edge, resolution, prob_labels, labels, cluster_sum_U, cluster_sum_I
    )
    time_cost = time.time() - time_begin
    user_clusters_co = multi_label[:num_user]
    item_clusters_co = multi_label[num_user:]
    print('type',type(user_clusters_co), len(user_clusters_co))
    user_clusters, num_user_clusters, user_count = map_to_consecutive_integers_multi(user_clusters_co)
    item_clusters, num_item_clusters, item_count = map_to_consecutive_integers_multi(item_clusters_co)
    mylog.write('Overlap Ratio user clusters:', user_count / num_user, 'item clusters:', item_count / num_item)
    mylog.write('increase nodes:', user_count - num_user)

    return time_cost, user_clusters, item_clusters, num_user_clusters, num_item_clusters, user_clusters_co, item_clusters_co
