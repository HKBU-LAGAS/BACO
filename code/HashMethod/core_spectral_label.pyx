# cython: language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as preinc

@cython.boundscheck(False)
@cython.wraparound(False)
def spectral_label_core(
        long num_user,
        long num_item,
        cnp.ndarray[long] nodes,
        cnp.ndarray[long] indptr,
        cnp.ndarray[long] indices,
        cnp.ndarray[float] data,
        cnp.ndarray[float] deg,
        float sum_edge,
        float resolution,
        cnp.ndarray[float] edge_weight,
        cnp.ndarray[long] labels,
        cnp.ndarray[float] cluster_sum_U,
        cnp.ndarray[float] cluster_sum_I,
        long update_opt,
):
    cdef long i, idx, x, l, y, start, end, label_target, label_best
    cdef float value, max_val
    cdef unordered_set[long] label_set  # 使用C++ unordered_set
    cdef unordered_set[long].iterator it  # 声明迭代器

    for i in range(nodes.size):
        x = nodes[i]
        start = indptr[x]
        end = indptr[x + 1]
        label_set.clear()

        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            l = labels[y]
            # value = data[idx] - resolution * deg[x] * deg[y] / sum_edge
            label_set.insert(l)
            edge_weight[l] += data[idx]

        # Find max value label and
        max_val = -np.inf if update_opt == 0 else 0
        label_best = labels[x]
        is_user = x < num_user
        is_item = x >= num_user
        old_value = update_opt * (edge_weight[label_best] - resolution * deg[x] / sum_edge * (is_user * cluster_sum_I[label_best] + is_item * cluster_sum_U[label_best]))
        it = label_set.begin()
        while it != label_set.end():
            label_target = deref(it)
            value = edge_weight[label_target] - resolution * deg[x] / sum_edge * (is_user * cluster_sum_I[label_target] + is_item * cluster_sum_U[label_target]) - old_value
            if value > max_val:
                max_val = value
                label_best = label_target
            edge_weight[label_target] = 0.0
            preinc(it)
        if label_best != labels[x]:
            if is_user:
                cluster_sum_U[label_best] += deg[x]
                cluster_sum_U[labels[x]] -= deg[x]
            else:
                cluster_sum_I[label_best] += deg[x]
                cluster_sum_I[labels[x]] -= deg[x]
            labels[x] = label_best
