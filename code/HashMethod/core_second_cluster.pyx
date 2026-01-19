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
# from libcpp.set cimport set
@cython.boundscheck(False)
@cython.wraparound(False)
def Second_Cluster_core(
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
):
    cdef int_or_long i, idx, x, l, y, start, end, label_target, label_best
    cdef float_or_double value, max_val
    cdef set user_label = set(labels[:num_user])
    cdef list multi_label = [[labels[_]] for _ in range(num_user + num_item)]  # 假设n_node=num_user+num_item
    for x in range(num_user):
        # label_set.clear()
        label_set = set()
        start = indptr[x]
        end = indptr[x + 1]

        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            l = labels[y]
            if l not in user_label: continue
            label_set.add(l)
            # if user_label.count(l) == 0: continue
            # label_set.insert(l)
            edge_weight[l] += data[idx]

        # Find max value label and
        max_val = -np.inf
        label_now = labels[x]
        label_best = labels[x]
        for label_target in label_set:
            value = edge_weight[label_target] - resolution * deg[x] / sum_edge * (cluster_sum_I[label_target] - cluster_sum_I[label_now])
            if label_target != label_now and value > max_val:
                max_val = value
                label_best = label_target
            edge_weight[label_target] = 0.0
        if label_best != label_now:
            multi_label[x] = [label_now, label_best]

    return multi_label
