# cython: language_level=3
# distutils: language = c++
ctypedef fused int_or_long:
    int
    long
ctypedef fused float_or_double:
    float
    double
cimport cython
import numpy as np
# cimport numpy as cnp
# from libcpp.unordered_set cimport unordered_set
# from cython.operator cimport dereference as deref, preincrement as preinc
from libcpp.set cimport set
@cython.boundscheck(False)
@cython.wraparound(False)
def spectral_label_core(
        int_or_long num_user,
        int_or_long num_item,
        int_or_long[:] nodes,
        int_or_long[:] indptr,
        int_or_long[:] indices,
        float_or_double[:] data,
        float_or_double[:] deg,
        float_or_double sum_edge,
        float_or_double resolution,
        float_or_double[:] edge_weight,
        int_or_long[:] labels,
        float_or_double[:] cluster_sum_U,
        float_or_double[:] cluster_sum_I,
        int_or_long update_opt,
):
    cdef int_or_long i, idx, x, l, y, start, end, label_target, label_best
    cdef float_or_double value, max_val, temp_r
    # cdef unordered_set[long] label_set  # 使用C++ unordered_set
    # cdef unordered_set[long].iterator it  # 声明迭代器
    cdef set[int_or_long] label_set = () # 使用C++ set

    for i in range(nodes.size):
        x = nodes[i]
        start = indptr[x]
        end = indptr[x + 1]
        label_set.clear()

        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            l = labels[y]
            label_set.insert(l)
            edge_weight[l] += data[idx]

        # Find max value label and
        max_val = -np.inf if update_opt == 0 else 0
        label_best = labels[x]
        is_user = x < num_user
        is_item = x >= num_user
        old_value = 0
        if update_opt:
            old_value = (edge_weight[label_best] - resolution * deg[x] / sum_edge * (is_user * cluster_sum_I[label_best] + is_item * cluster_sum_U[label_best]))
        temp_r = resolution * deg[x] / sum_edge
        for label_target in label_set:
            if is_user:
                value = edge_weight[label_target] - temp_r * cluster_sum_I[label_target] - old_value
            else:
                value = edge_weight[label_target] - temp_r * cluster_sum_U[label_target] - old_value

            if value > max_val:
                max_val = value
                label_best = label_target
            edge_weight[label_target] = 0.0
        if label_best != labels[x]:
            if is_user:
                cluster_sum_U[label_best] += deg[x]
                cluster_sum_U[labels[x]] -= deg[x]
            else:
                cluster_sum_I[label_best] += deg[x]
                cluster_sum_I[labels[x]] -= deg[x]
            labels[x] = label_best
