import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class UserItemRatingDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index

        # Extract the unique user and item IDs from the edge_index
        self.user_ids = np.unique(edge_index[0])
        self.item_ids = np.unique(edge_index[1])
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

    def __len__(self):
        return self.edge_index.shape[-1]

    def __getitem__(self, idx):
        user_id = self.edge_index[0, idx].item()
        item_id = self.edge_index[1, idx].item()

        return (user_id, item_id)


class UserItemRatingDatasetv2(Dataset):
    def __init__(self, edge_index, user_features, item_features, ratings):
        self.edge_index = edge_index
        self.user_features = user_features
        self.item_features = item_features
        self.ratings = ratings

        # Extract the unique user and item IDs from the edge_index
        self.user_ids = np.unique(edge_index[0])
        self.item_ids = np.unique(edge_index[1])
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

    def __len__(self):
        return self.edge_index.shape[-1]

    def __getitem__(self, idx):
        user_id = self.edge_index[0, idx].item()
        item_id = self.edge_index[1, idx].item()
        user_feature = self.user_features[user_id, :]
        item_feature = self.item_features[item_id, :]
        rating = self.ratings[idx].item()

        return (user_id, item_id, user_feature, item_feature, rating)


class DatasetLoader:
    def __init__(self, dataset):
        train_file_name = f"datasets/{dataset}-processed/train.pt"
        self.train_set = UserItemRatingDataset(torch.load(train_file_name,weights_only=False))

        val_file_name = f"datasets/{dataset}-processed/val.pt"
        self.val_set = UserItemRatingDataset(torch.load(val_file_name,weights_only=False))

        test_file_name = f"datasets/{dataset}-processed/test.pt"
        self.test_set = UserItemRatingDataset(torch.load(test_file_name,weights_only=False))

    def get_datasets(self):
        return self.train_set, self.val_set, self.test_set


