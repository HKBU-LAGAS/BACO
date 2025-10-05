from sys import implementation
import torch, os, random, math, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from scipy.linalg import clarkson_woodruff_transform
from collections import Counter

def normalize_adjacency(adj):
    row_diag = np.asarray(1.0 / np.sqrt(adj.sum(axis=1))).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(adj.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    n_rows, n_cols = adj.shape
    r = sp.dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
    c = sp.dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
    normalized_data = r * adj * c
    return normalized_data

def no_bias_adjacency(adj):
    m = adj.nnz
    row_sum = np.array(adj.sum(axis=1)).squeeze()
    col_sum = np.array(adj.sum(axis=0)).squeeze()
    new_adj = adj.copy()
    rows, cols = new_adj.nonzero()
    corrections = (row_sum[rows] * col_sum[cols]) / (m)
    new_adj = new_adj.tocsr()
    new_adj.data -= corrections
    return new_adj


def sparse_argmax_csr(A):
    labels = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        start, end = A.indptr[i], A.indptr[i+1]
        if start == end:
            labels[i] = 0  # 处理全零行（按需修改）
        else:
            max_pos = np.argmax(A.data[start:end])
            labels[i] = A.indices[start + max_pos]
    return labels

def build_block(mat):
    mat_coo = mat.tocoo()
    n_user, n_item = mat.shape
    upper_rows = mat_coo.row
    upper_cols = mat_coo.col + n_user
    lower_rows = mat_coo.col + n_user
    lower_cols = mat_coo.row
    all_rows = np.concatenate([upper_rows, lower_rows])
    all_cols = np.concatenate([upper_cols, lower_cols])
    all_data = np.concatenate([mat_coo.data, mat_coo.data])
    return sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(n_user+n_item, n_user+n_item)).tocsr()
