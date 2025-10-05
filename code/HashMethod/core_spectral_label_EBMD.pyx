# cython: language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as preinc

@cython.boundscheck(False)
@cython.wraparound(False)
def spectral_label_EBMD_core(
        int num_user,
        int num_item,
        cnp.ndarray[long] nodes,
        cnp.ndarray[long] indptr,
        cnp.ndarray[long] indices,
        cnp.ndarray[double] data,
        cnp.ndarray[double] deg,
        double sum_edge,
        double resolution,
        cnp.ndarray[double] tmp_val,
        cnp.ndarray[double] tmp_cluster_edge,
        cnp.ndarray[double] cluster_edge,
        cnp.ndarray[long] labels,
        cnp.ndarray[double] cluster_sum_U,
        cnp.ndarray[double] cluster_sum_I,
):
    cdef long i, idx, x, l, y, start, end, label_target, label_best, Cu, Cv
    cdef double max_value, gamma, prob
    cdef unordered_set[long] label_set  # 使用C++ unordered_set
    cdef unordered_set[long].iterator it  # 声明迭代器
    global_density = sum_edge / num_user / num_item
    for i in range(nodes.size):
        x = nodes[i]
        start = indptr[x]
        end = indptr[x + 1]
        label_set.clear()

        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            l = labels[y]
            tmp_cluster_edge[l] += data[idx]
            tmp_val[l] += deg[y]
            label_set.insert(l)

        # Find max value label and
        max_value = -np.inf
        label_best = labels[x]
        ori_edge = tmp_cluster_edge[labels[x]]
        add_edge = 0
        it = label_set.begin()
        is_user = x < num_user
        is_item = x >= num_user
        while it != label_set.end():
            label_target = deref(it)

            Cu = cluster_sum_U[label_target] + is_user
            Cv = cluster_sum_I[label_target] + is_item
            local_density = (cluster_edge[label_target] + tmp_cluster_edge[label_target]) / Cu / Cv
            gamma = local_density - global_density
            prob = gamma * tmp_cluster_edge[label_target] - resolution * gamma * gamma * deg[x] / sum_edge * (is_user * cluster_sum_I[label_target] + is_item * cluster_sum_U[label_target])
            if prob > max_value:
                max_value = prob
                label_best = label_target
                add_edge = tmp_cluster_edge[label_target]
                # print('gamma:',gamma, 'prob:',prob)

            tmp_cluster_edge[label_target] = 0
            tmp_val[label_target] = 0
            preinc(it)

        if label_best != labels[x]:
            cluster_edge[labels[x]] -= ori_edge
            cluster_edge[label_best] += add_edge
            if x < num_user:
                cluster_sum_U[labels[x]] -= deg[x]
                cluster_sum_U[label_best] += deg[x]
            else:
                cluster_sum_I[labels[x]] -= deg[x]
                cluster_sum_I[label_best] += deg[x]
            labels[x] = label_best

