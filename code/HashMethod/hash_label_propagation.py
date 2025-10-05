import time
import numpy as np
from spectral_label_kit import build_block
from HashMethod.utils import map_to_consecutive_integers, double_map_to_consecutive_integers
from HashMethod.core_label_propagation import label_propagation_core
def label_propagation_hash(
    biadjacency,
    num_users_c,
    num_items_c,
    n_iter=10,
    alpha=0.75,
    random_state=2025,
    mylog=None,
    Secondary_Clusters=False,
):
    print('input num cluster:',num_users_c, num_items_c)

    time_begin = time.time()
    rng = np.random.default_rng(random_state)
    num_user, num_item = biadjacency.shape
    n_node = num_user + num_item
    max_cluster = num_user + num_item
    labels = np.arange(max_cluster, dtype=np.int64)
    prob_labels = np.zeros(max_cluster, dtype=np.float64)
    adj = build_block(biadjacency)
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    data = adj.data.astype(np.float64)

    for t in range(n_iter):
        # Update labels for each node in random order
        nodes = rng.permutation(n_node)
        randoms = rng.random(n_node)
        update_flags = (randoms < alpha).astype(np.uint8)
        # for x in nodes:
        #     if not update_flags[x]: continue
        #     prob_labels = {}
        #     for i in range(adj.indptr[x], adj.indptr[x + 1]):
        #         y = adj.indices[i]
        #         value = adj.data[i]
        #         l = labels[y]
        #         prob_labels[l] = prob_labels.get(l, 0) + value
        #     max_value = -np.inf
        #     label_best = labels[x]
        #     if prob_labels:
        #         for label_target, value in prob_labels.items():
        #             if value > max_value:
        #                 max_value = value
        #                 label_best = label_target
        #         labels[x] = label_best
        label_propagation_core(nodes, update_flags, indptr, indices, data, prob_labels, labels)
        print('labels num:', len(set(labels)), 'target:', num_users_c + num_items_c)
    if Secondary_Clusters == False :
        time_cost = time.time() - time_begin
        user_clusters_co = labels[:num_user]
        item_clusters_co = labels[num_user:]
        user_clusters, num_user_clusters = map_to_consecutive_integers(user_clusters_co)
        item_clusters, num_item_clusters = map_to_consecutive_integers(item_clusters_co)
        user_clusters_co, item_clusters_co, num_share_clusters = double_map_to_consecutive_integers(user_clusters_co, item_clusters_co)
    else :
        from collections import defaultdict
        from HashMethod.utils import map_to_consecutive_integers_multi
        multi_label = [[labels[_]] for _ in range(n_node)]
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