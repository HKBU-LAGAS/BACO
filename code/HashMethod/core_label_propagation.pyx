# cython: language_level=3
# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as preinc

@cython.boundscheck(False)
@cython.wraparound(False)
def label_propagation_core(
        cnp.ndarray[long] nodes,
        cnp.uint8_t[:] update_flags,
        cnp.ndarray[long] indptr,
        cnp.ndarray[long] indices,
        cnp.ndarray[double] data,
        cnp.ndarray[double] prob_map,
        cnp.ndarray[long] labels,
):
    cdef long i, idx, x, l, y, start, end, label_target, label_best
    cdef double value, max_val
    cdef unordered_set[long] label_set  # 使用C++ unordered_set
    cdef unordered_set[long].iterator it  # 声明迭代器

    for i in range(nodes.size):
        x = nodes[i]
        if not update_flags[i]:
            continue  # 跳过不更新的节点
        start = indptr[x]
        end = indptr[x + 1]
        label_set.clear()
        # Aggregate values for each neighbor's label
        for idx in range(start, end):
            y = indices[idx]
            value = data[idx]
            l = labels[y]
            label_set.insert(l)
            prob_map[l] += value

        # Find max value label and
        max_val = -np.inf
        label_best = labels[x]
        it = label_set.begin()
        while it != label_set.end():
            label_target = deref(it)
            if prob_map[label_target] > max_val:
                max_val = prob_map[label_target]
                label_best = label_target
            prob_map[label_target] = 0.0
            preinc(it)

        labels[x] = label_best
