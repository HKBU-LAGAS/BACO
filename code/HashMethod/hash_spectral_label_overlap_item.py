import time
import numpy as np
from HashMethod.utils import map_to_consecutive_integers_multi, build_block
from HashMethod.core_spectral_label import spectral_label_core

def spectral_label_overlap_item_hash(
    biadjacency,
    num_users_c,
    num_items_c,
    resolution=1.0,
    n_iter=5,
    random_state=2025,
    weight_scheme={'user': 'degree', 'item': 'degree'},
    mylog=None,
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
    if weight_scheme['user'] == 'CPM':  # 1
        user_value = np.ones(num_user, dtype=np.float64)
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
    # cluster_sum_U = np.concatenate([user_weight, np.zeros(num_item, dtype=np.float64)])
    # cluster_sum_I = np.concatenate([np.zeros(num_user, dtype=np.float64), item_weight])
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

    multi_label = [[labels[_]] for _ in range(n_node)]
    item_label = set(labels[num_user:])
    for id in range(num_item):
        x = num_user + id
        label_set = set()
        # maintain mini community
        for i in range(adj.indptr[x], adj.indptr[x + 1]):
            y = adj.indices[i]
            l = labels[y]
            if l not in item_label: continue
            prob_labels[l] += adj.data[i]
            label_set.add(l)
        max_value = None
        label_now = labels[x]
        label_best = labels[x]
        for label_target in label_set:
            now_value = prob_labels[label_target] - resolution * deg[x] / sum_edge * (cluster_sum_I[label_target] - cluster_sum_I[label_now])
            if label_target != label_now:
                if (max_value is None) or (now_value > max_value):
                    max_value = now_value
                    label_best = label_target
            prob_labels[label_target] = 0
        if label_now == label_best: continue
        top_labels = [label_now, label_best]
        multi_label[x] = top_labels


    time_cost = time.time() - time_begin
    user_clusters_co = multi_label[:num_user]
    item_clusters_co = multi_label[num_user:]
    user_clusters, num_user_clusters, user_count = map_to_consecutive_integers_multi(user_clusters_co)
    item_clusters, num_item_clusters, item_count = map_to_consecutive_integers_multi(item_clusters_co)
    mylog.write('Overlap Ratio user clusters:', user_count / num_user, 'item clusters:', item_count / num_item)
    mylog.write('increase nodes:', user_count - num_user + item_count - num_item)

    return time_cost, user_clusters, item_clusters, num_user_clusters, num_item_clusters, user_clusters_co, item_clusters_co