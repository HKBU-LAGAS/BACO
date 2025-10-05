import time
import numpy as np
from sklearn.cluster import SpectralBiclustering
from HashMethod.utils import map_to_consecutive_integers, double_map_to_consecutive_integers

class CustomSBC(SpectralBiclustering):
    def _fit(self, X):
        # import time
        from sklearn.cluster._bicluster import _bistochastic_normalize,_scale_normalize,_log_normalize
        # start_time = time.time()
        n_sv = self.n_components
        if self.method == "bistochastic":
            normalized_data = _bistochastic_normalize(X)
            n_sv += 1
        elif self.method == "scale":
            normalized_data, _, _ = _scale_normalize(X)
            n_sv += 1
        elif self.method == "log":
            normalized_data = _log_normalize(X)
        # print('normalized_data !')
        # print('time cost of normalization:', time.time() - start_time)
        # start_time = time.time()
        n_discard = 0 if self.method == "log" else 1
        u, v = self._svd(normalized_data, n_sv, n_discard)
        ut = u.T
        vt = v.T
        # print('svd done !')
        # print('time cost of svd:', time.time() - start_time)
        # start_time = time.time()
        try:
            n_row_clusters, n_col_clusters = self.n_clusters
        except TypeError:
            n_row_clusters = n_col_clusters = self.n_clusters

        best_ut = self._fit_best_piecewise(ut, self.n_best, n_row_clusters)

        best_vt = self._fit_best_piecewise(vt, self.n_best, n_col_clusters)
        # print('fit best done !')
        # print('time cost of fit best:', time.time() - start_time)
        # start_time = time.time()
        self.row_labels_ = self._project_and_cluster(X, best_vt.T, n_row_clusters)

        self.column_labels_ = self._project_and_cluster(X.T, best_ut.T, n_col_clusters)
        # print('project and cluster done !')
        # print('time cost of project and cluster:', time.time() - start_time)

        # self.rows_ = np.vstack(
        #     [
        #         self.row_labels_ == label
        #         for label in range(n_row_clusters)
        #         for _ in range(n_col_clusters)
        #     ]
        # )
        # self.columns_ = np.vstack(
        #     [
        #         self.column_labels_ == label
        #         for _ in range(n_row_clusters)
        #         for label in range(n_col_clusters)
        #     ]
        # )

def SpectralBC_hash(
    biadjacency,
    resolution,
    num_users_c,
    num_items_c,
):
    num_share_clusters = resolution
    print('init num: ', num_users_c, num_items_c, num_share_clusters)
    time_begin = time.time()
    # 1. 找全零行/列（在CSR下直接用sparse sum）
    row_sum = np.array(biadjacency.sum(axis=1)).ravel()  # shape (n_users,)
    col_sum = np.array(biadjacency.sum(axis=0)).ravel()  # shape (n_items,)
    nonzero_row_mask = row_sum != 0
    nonzero_col_mask = col_sum != 0

    zero_row_indices = np.where(~nonzero_row_mask)[0]
    zero_col_indices = np.where(~nonzero_col_mask)[0]
    nonzero_row_indices = np.where(nonzero_row_mask)[0]
    nonzero_col_indices = np.where(nonzero_col_mask)[0]

    # 2. 用稀疏切片获取子矩阵
    sub_biadjacency = biadjacency[nonzero_row_mask][:, nonzero_col_mask]
    print("sub_biadjacency shape:", sub_biadjacency.shape)
    # 可以检查sum是否都不是0（可选）
    assert sub_biadjacency.shape[0] > 0 and sub_biadjacency.shape[
        1] > 0, "All rows or columns are zero after filtering!"

    # Spectral Co-clustering
    model = CustomSBC(
        n_clusters=num_share_clusters,  # 关键参数：指定共聚类数量
        random_state=2025,  # 固定随机种子保证可复现
    )
    model.fit(sub_biadjacency)
    # print('fitting done !')
    time_cost = time.time() - time_begin

    user_clusters_nonzero = model.row_labels_
    item_clusters_nonzero = model.column_labels_

    n_total_users = biadjacency.shape[0]
    n_total_items = biadjacency.shape[1]
    user_clusters = np.full(n_total_users, -1, dtype=int)
    item_clusters = np.full(n_total_items, -1, dtype=int)
    user_clusters[nonzero_row_indices] = user_clusters_nonzero
    item_clusters[nonzero_col_indices] = item_clusters_nonzero

    curr_user_label = num_share_clusters
    for idx in zero_row_indices:
        user_clusters[idx] = curr_user_label
        curr_user_label += 1

    curr_item_label = num_share_clusters
    for idx in zero_col_indices:
        item_clusters[idx] = curr_item_label
        curr_item_label += 1

    user_clusters_sep, num_users_c = map_to_consecutive_integers(user_clusters)
    item_clusters_sep, num_items_c = map_to_consecutive_integers(item_clusters)
    print('process num: ', num_users_c, num_items_c, num_share_clusters)
    # spectral_label_kit.analyze_label_distribution(np.concatenate([user_clusters_ori, item_clusters_ori]),
    #                                               save_dir='./results/pic', file_prefix='spectral')
    return time_cost, user_clusters_sep, item_clusters_sep, num_users_c, num_items_c, user_clusters, item_clusters
