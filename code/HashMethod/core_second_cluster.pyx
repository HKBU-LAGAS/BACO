# cython: language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as preinc

@cython.boundscheck(False)
@cython.wraparound(False)
def Second_Cluster_core(
        int num_user,
        int num_item,
        cnp.ndarray[long] nodes,
        cnp.ndarray[long] indptr,
        cnp.ndarray[long] indices,
        cnp.ndarray[double] data,
        cnp.ndarray[double] deg,
        double sum_edge,
        double resolution,
        cnp.ndarray[double] edge_weight,
        cnp.ndarray[long] labels,
        cnp.ndarray[double] cluster_sum_U,
        cnp.ndarray[double] cluster_sum_I,
):
    cdef long i, idx, x, l, y, start, end, label_target, label_best
    cdef double value, max_val
    cdef set user_label = set([labels[i] for i in range(num_user)])
    cdef list multi_label = [[labels[_]] for _ in range(num_user + num_item)]  # 假设n_node=num_user+num_item
    for x in range(num_user):
        label_set = set()
        start = indptr[x]
        end = indptr[x + 1]

        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            l = labels[y]
            if l not in user_label: continue
            label_set.add(l)
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