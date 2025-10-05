import torch
import torch.nn as nn
from .baserec import BaseRecommender
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class LightGCN(BaseRecommender):
    def get_norm_adj_mat(self):
        """Get the normalized interaction matrix of users and items.
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        """

        inter_M = self.interaction_matrix
        users, items = inter_M.nonzero()
        rows_u2i = users
        cols_u2i = items +self.n_users

        inter_M_t = self.interaction_matrix.T
        items_t, users_t = inter_M_t.nonzero()
        rows_i2u = items_t + self.n_users
        cols_i2u = users_t

        all_rows = np.concatenate([rows_u2i, rows_i2u])
        all_cols = np.concatenate([cols_u2i, cols_i2u])
        A = sp.coo_matrix(
            (np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items)
        )
        A_csr = A.tocsr()
        degrees = np.array(A_csr.sum(axis=1)).flatten()
        diag = 1.0 / np.sqrt(degrees + 1e-7)
        rows, cols = A.row, A.col
        scaling = diag[rows] * diag[cols]  # 每个元素乘以 sqrt(d_i*d_j)
        data_norm = A.data * scaling
        L = sp.coo_matrix((data_norm, (rows, cols)), shape=A.shape)
        indices = torch.from_numpy(np.vstack([L.row, L.col])).long()
        values = torch.from_numpy(L.data).float()
        # SparseL = torch.sparse.FloatTensor(indices, values, torch.Size(L.shape))
        SparseL = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=torch.Size(L.shape),
            dtype=torch.float32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # # build adj matrix
        # A = sp.sparse.dok_matrix(
        #     (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        # )
        # row_indices, col_indices = inter_M.nonzero()
        # row_indices_t, col_indices_t = inter_M_t.nonzero()
        #
        # data_dict = dict(
        #     zip(zip(row_indices, col_indices + self.n_users), [1] * inter_M.nnz)
        # )
        # data_dict.update(
        #     dict(
        #         zip(
        #             zip(row_indices_t + self.n_users, col_indices_t),
        #             [1] * inter_M_t.nnz,
        #         )
        #     )
        # )
        # A.update(data_dict)
        # # norm adj matrix
        # sumArr = (A > 0).sum(axis=1)
        # # add epsilon to avoid Devide by zero Warning
        # diag = np.array(sumArr.flatten())[0] + 1e-7
        # diag = np.power(diag, -0.5)
        # D = sp.sparse.diags(diag)
        # L = D * A * D
        # # covert norm_adj matrix to tensor
        # L = sp.sparse.coo_matrix(L)
        # row = L.row
        # col = L.col
        # i = torch.LongTensor(np.array([row, col]))
        # data = torch.FloatTensor(L.data)
        # SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        if self.hash_type == 'Single':
            user_embeddings = self.user_emb_table(self.user_hashed_ids)
            item_embeddings = self.item_emb_table(self.item_hashed_ids)
        elif self.hash_type == 'Double':
            user_embeddings = self.user_emb_table(self.user_hashed_ids)
            item_embeddings = self.item_emb_table(self.item_hashed_ids)
            user_embeddings = torch.sum(user_embeddings, dim=-2)
            item_embeddings = torch.sum(item_embeddings, dim=-2)
        elif self.hash_type == 'Multi':
            # user_embeddings = self.user_emb_table(self.user_hashed_ids)
            # item_embeddings = self.item_emb_table(self.item_hashed_ids)
            user_embeddings = self.user_emb_table(self.user_hashed_ids, self.user_offsets)
            item_embeddings = self.item_emb_table(self.item_hashed_ids, self.item_offsets)
        else :
            raise NotImplementedError(f'No such hash type: {self.hash_type} !')

        # print('emb shape:',user_embeddings.shape,item_embeddings.shape)
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings

    def propagate(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def __init__(
            self,
            user_vocab_size: int,
            item_vocab_size: int,
            embedding_dim: int,
            user_hashed_ids,
            item_hashed_ids,
            biadjacency=None,
            hash_type="None",
            num_layers=2,
            multi_mode='mean'
    ):
        super(LightGCN, self).__init__()
        if hash_type == "None":
            raise ValueError("calc_type must not be None")

        self.embedding_dim = embedding_dim
        # define layers
        self.interaction_matrix = biadjacency
        self.n_users = biadjacency.shape[0]
        self.n_items = biadjacency.shape[1]
        self.n_layers = num_layers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hash_type = hash_type
        if hash_type == "Multi":
            self.user_emb_table = nn.EmbeddingBag(
                num_embeddings=user_vocab_size,
                embedding_dim=embedding_dim,
                mode='mean'  # sum/mean/max
            )
            self.item_emb_table = nn.EmbeddingBag(
                num_embeddings=item_vocab_size,
                embedding_dim=embedding_dim,
                mode='mean'  # sum/mean/max
            )
            user_labels = [torch.tensor(labels) for labels in user_hashed_ids]
            self.user_hashed_ids = torch.cat(user_labels).to(device)
            self.user_offsets = torch.cumsum(torch.tensor([0] + [len(l) for l in user_labels[:-1]]), dim=0).to(device)
            item_labels = [torch.tensor(labels) for labels in item_hashed_ids]
            self.item_hashed_ids = torch.cat(item_labels).to(device)
            self.item_offsets = torch.cumsum(torch.tensor([0] + [len(l) for l in item_labels[:-1]]), dim=0).to(device)

        else:
            self.user_emb_table = nn.Embedding(user_vocab_size, embedding_dim)
            self.item_emb_table = nn.Embedding(item_vocab_size, embedding_dim)
            self.user_hashed_ids = torch.tensor(user_hashed_ids).to(device)
            self.item_hashed_ids = torch.tensor(item_hashed_ids).to(device)
        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(device)


    def forward(self, user_id, pos_item_id, neg_item_id, hash_type=None):
        user_all_embeddings, item_all_embeddings = self.propagate()
        user_id_embeddings = user_all_embeddings[user_id]
        pos_item_id_embeddings = item_all_embeddings[pos_item_id]
        neg_item_id_embeddings = item_all_embeddings[neg_item_id]

        pos_score = torch.sum(user_id_embeddings * pos_item_id_embeddings, dim=-1)

        neg_score = torch.sum(user_id_embeddings * neg_item_id_embeddings, dim=-1)

        return pos_score, neg_score

    def get_scores(self, hash_type, device, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.propagate()
        user_id_embeddings = user_all_embeddings[user_id]
        item_id_embeddings = item_all_embeddings[item_id]

        scores = user_id_embeddings @ item_id_embeddings.t()
        return scores

    def get_embedding(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.propagate()
        user_id_embeddings = user_all_embeddings[user_id]
        item_id_embeddings = item_all_embeddings[item_id]
        return user_id_embeddings, item_id_embeddings
