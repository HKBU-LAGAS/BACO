import time
import psutil
from omegaconf import OmegaConf
from re import error
import json
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
from log import Logger
from torch.utils.data import DataLoader
import utils
import HashMethod
from utils import (
    training_graph,
    graphClustering,
    candidate_set,
    map_to_consecutive_integers,
    count_parameters,
)
from data import DatasetLoader
from training.train_test_id_only import train_and_evaluate
from loss import BPRLoss
from data import (UserItemRatingDataset,UserItemRatingDatasetv2)
import os
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
# import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(config):
    # Hyperparameters from command line arguments
    config_dict = OmegaConf.to_container(config, resolve=True)
    dataset = config.dataset.name
    embedding_dim = config.dataset.embed_dim
    num_epochs = config.epoch
    batch_size = config.dataset.bs
    learning_rate = config.dataset.lr
    weight_decay = config.dataset.wd
    patience = config.dataset.patience
    k = config.dataset.k
    hash_type = config.hash_type
    resolution = config.dataset.resolution
    loss = config.dataset.loss
    target = config.model._target_
    model_name = target.split(".")[-1]
    num_layers = config.model.num_layers

    # model list
    GraphHash_list = ["full", "random", "frequency", "lsh",
                      'EBMD', "spectral_label_Test",
                      "GraphHash", 'label_propagation', 'Leiden', 'Louvain',
                      'SBC', "SCC",
                      ]
    DoubleHash_list = ["double", "double_frequency", "double_graph",]
    Multi_list = ["BACO", "spectral_label_overlap_item", "spectral_label_overlap_double"]

    # Create a run name using the hyperparameters
    if hash_type in ['spectral_label_Test']:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}"
    elif hash_type in ['SBC', 'SCC']:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}"
    elif hash_type in ['Louvain', 'Leiden']:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}_mod_{config.mod_type}"
    elif hash_type in ["full", 'random', 'frequency', 'lsh']:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}"
    elif hash_type in ['GraphHash', 'double_graph']:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}"
    elif hash_type in Multi_list:
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}"
    else :
        run_name = f"{model_name}_{dataset}_hash_{hash_type}_res_{config.resolution}"
    # Initialize log
    record_dir = config.save_dir
    os.makedirs(record_dir, exist_ok=True)
    if config.Secondary_Clusters == True:
        run_name += '_Secondary'
    mylog = Logger(record_dir, run_name + '.txt')
    mylog.write(str(config_dict) + '\n')
    print('dataset:', dataset)
    mylog.write_summary({'dataset': dataset})
    mylog.write_summary({'hash-type': config.hash_type})

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.vis_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init memory
    torch.cuda.reset_peak_memory_stats(device)  # reset GPU memory
    start_cpu_mem = psutil.Process().memory_info().rss  # reset CPU memory
    # read data
    data = DatasetLoader(dataset=dataset)
    train_set, val_set, test_set = data.get_datasets()
    uid_set = set()
    iid_set = set()
    for user_id, item_id, *additional_info in train_set:
        uid_set.add(user_id)
        iid_set.add(item_id)
    # num_total_users = len(uid_set)
    # num_total_items = len(iid_set)
    num_total_users = config.dataset.num_total_users
    num_total_items = config.dataset.num_total_items
    print(num_total_users, num_total_items)

    # build the relevant item set for each user
    train_candidates, val_candidates, test_candidates = (
        candidate_set(train_set),
        candidate_set(val_set),
        candidate_set(test_set),
    )
    mylog.write_summary(
        {
            "Number of train users": len(train_candidates),
            "Number of val users": len(val_candidates),
            "Number of test users": len(test_candidates),
        }
    )

    # hashing:
    # Generate adjacency
    biadjacency = training_graph(train_set=train_set, num_total_users=num_total_users, num_total_items=num_total_items,)

    # Louvain Cluster
    (user_clusters, item_clusters, num_users_clusters, num_items_clusters, graph_cluster_cost) \
        = graphClustering(biadjacency=biadjacency, resolution=resolution)

    standard_hash_parameter = tensor_size(user_clusters) + tensor_size(item_clusters)
    mylog.write("Standard Hash parameter:", standard_hash_parameter)
    extra_hash_parameter = 0
    calc_type = 'None'
    cluster_cost = -1

    if hash_type in GraphHash_list:
        calc_type = 'Single'
        if hash_type == "full":
            cluster_cost = 0
            user_clusters = np.arange(num_total_users)
            item_clusters = np.arange(num_total_items)
            label_row = user_clusters
            label_col = item_clusters
            num_users_clusters, num_items_clusters = num_total_users, num_total_items
        elif hash_type == "GraphHash":
            (user_clusters, item_clusters, num_users_clusters, num_items_clusters, graph_cluster_cost) \
                = graphClustering(biadjacency=biadjacency, resolution=config.resolution)
            label_row = user_clusters
            label_col = item_clusters
            user_clusters = map_to_consecutive_integers(user_clusters)
            item_clusters = map_to_consecutive_integers(item_clusters)
            cluster_cost = graph_cluster_cost
        elif hash_type == "random":
            cluster_cost = time.time()
            user_clusters = np.arange(num_total_users) % num_users_clusters
            item_clusters = np.arange(num_total_items) % num_items_clusters
            label_row = user_clusters
            label_col = item_clusters
            cluster_cost = time.time() - cluster_cost
        elif hash_type == "frequency":
            cluster_cost = time.time()
            user_clusters, item_clusters = utils.frequency_hash(
                biadjacency=biadjacency,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )
            label_row = user_clusters
            label_col = item_clusters
            cluster_cost = time.time() - cluster_cost
        elif hash_type == "lsh":
            cluster_cost = time.time()
            biadjacency_tensor = torch.tensor(biadjacency.toarray())

            user_clusters, item_clusters, num_users_clusters, num_items_clusters = utils.lsh(
                user_features=biadjacency_tensor,
                item_features=biadjacency_tensor.T,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
            )
            label_row = user_clusters
            label_col = item_clusters
            cluster_cost = time.time() - cluster_cost
        elif hash_type == "SCC": # Spectral Co-clustering
            cluster_cost,user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.SpectralCC_hash(
                biadjacency,
                config.resolution,
                num_users_clusters,
                num_items_clusters,
            )
        elif hash_type == "SBC": # Spectral Bi-clustering
            cluster_cost,user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.SpectralBC_hash(
                biadjacency,
                config.resolution,
                num_users_clusters,
                num_items_clusters,
            )
        elif hash_type == "label_propagation":
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.label_propagation_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                n_iter=config.n_iter,
                alpha=config.resolution,
                mylog=mylog,
                Secondary_Clusters=config.Secondary_Clusters,
            )
        elif hash_type == "Leiden":
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.Leiden_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                mod_type=config.mod_type,
                mylog=mylog,
                Secondary_Clusters=config.Secondary_Clusters,
            )
        elif hash_type == "Louvain":
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.Louvain_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                mod_type=config.mod_type,
                mylog=mylog,
                Secondary_Clusters=config.Secondary_Clusters,
            )
        elif hash_type == "EBMD":
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.spectral_label_EBMD_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                n_iter=config.n_iter,
            )
        elif hash_type == "spectral_label_Test":
            weight_scheme = {'user': 'degree', 'item': 'degree'}
            if config.user_weight == None or config.item_weight == None:
                raise ValueError('weight_type is None!')
            else:
                weight_scheme['user'] = config.user_weight
                weight_scheme['item'] = config.item_weight
            print('weight_scheme:', weight_scheme)
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.spectral_label_Test_hash(
                biadjacency,
                n_iter=config.n_iter,
                resolution=config.resolution,
                weight_scheme=weight_scheme,
            )
        else :
            raise NotImplementedError(f'Single type is not implement {hash_type}!')
    elif hash_type in DoubleHash_list:
        calc_type = 'Double'
        if hash_type == "double_graph":
            cluster_cost = time.time()
            user_clusters, item_clusters = utils.double_graph_hash(
                biadjacency,
                config.resolution,
                num_users_clusters,
                num_items_clusters,
                num_total_users,
                num_total_items,
            )
            cluster_cost = time.time() - cluster_cost

        elif hash_type == "double":
            cluster_cost = time.time()
            user_clusters, item_clusters = utils.double_hash(
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )
            cluster_cost = time.time() - cluster_cost

        elif hash_type == "double_frequency":
            cluster_cost = time.time()
            user_clusters, item_clusters = utils.double_frequency_hash(
                biadjacency=biadjacency,
                num_users_clusters=num_users_clusters,
                num_items_clusters=num_items_clusters,
                num_total_users=num_total_users,
                num_total_items=num_total_items,
            )
            cluster_cost = time.time() - cluster_cost

        else :
            raise NotImplementedError(f'Double type is not implement {hash_type}!')

    elif hash_type in Multi_list:
        calc_type = 'Multi'
        if hash_type == "BACO":
            weight_scheme = {'user': 'degree', 'item': 'CPM'}
            if config.user_weight == None or config.item_weight == None:
                raise ValueError('weight_type is None!')
            else:
                weight_scheme['user'] = config.user_weight
                weight_scheme['item'] = config.item_weight
            print('weight_scheme:', weight_scheme)
            mylog.write_summary(weight_scheme)
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.spectral_label_overlap_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                n_iter=config.n_iter,
                weight_scheme=weight_scheme,
                mylog=mylog,
            )
        elif hash_type == "spectral_label_overlap_item":
            weight_scheme = {'user': 'degree', 'item': 'CPM'}
            if config.user_weight == None or config.item_weight == None:
                raise ValueError('weight_type is None!')
            else:
                weight_scheme['user'] = config.user_weight
                weight_scheme['item'] = config.item_weight
            print('weight_scheme:', weight_scheme)
            mylog.write_summary(weight_scheme)
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.spectral_label_overlap_item_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                n_iter=config.n_iter,
                weight_scheme=weight_scheme,
                mylog=mylog,
            )
        elif hash_type == "spectral_label_overlap_double":
            weight_scheme = {'user': 'degree', 'item': 'CPM'}
            if config.user_weight == None or config.item_weight == None:
                raise ValueError('weight_type is None!')
            else:
                weight_scheme['user'] = config.user_weight
                weight_scheme['item'] = config.item_weight
            print('weight_scheme:', weight_scheme)
            mylog.write_summary(weight_scheme)
            cluster_cost, user_clusters, item_clusters, num_users_clusters, num_items_clusters, label_row, label_col = HashMethod.spectral_label_overlap_double_hash(
                biadjacency,
                num_users_clusters,
                num_items_clusters,
                resolution=config.resolution,
                n_iter=config.n_iter,
                weight_scheme=weight_scheme,
                mylog=mylog,
            )
        else :
            raise NotImplementedError(f'Double type is not implement {hash_type}!')
    else :
        raise NotImplementedError(f'No such hash type: {hash_type} !')

    mylog.write(f'Cluster Time:{cluster_cost}')
    mylog.write_summary({'Cluster Time': cluster_cost})
    extra_hash_parameter = tensor_size(user_clusters) + tensor_size(item_clusters) - standard_hash_parameter
    mylog.write("Extra Hash parameter:", extra_hash_parameter)
    if config.Secondary_Clusters == True: calc_type = 'Multi'

    mylog.write_summary(
        {
            "Number of user clusters": num_users_clusters,
            "Number of item clusters": num_items_clusters,
        }
    )
    print(hash_type, 'cluster number {user item}:', num_users_clusters, num_items_clusters,)
    mylog.write(hash_type + ' cluster number:' + str(num_users_clusters)+' '+str(num_items_clusters)+'\n', is_terminal=0)

    # Initialize model, loss function, and optimizer
    if loss in ["BPR", "DAU", "iALS"]: # retrieval
        mylog.write_summary({"Number of Layer": num_layers})
        model = instantiate(
            config.model,
            user_vocab_size=num_users_clusters,
            item_vocab_size=num_items_clusters,
            biadjacency=biadjacency,
            user_hashed_ids=user_clusters,
            item_hashed_ids=item_clusters,
            hash_type=calc_type,
            embedding_dim=embedding_dim,
        ).to(device)

        num_parameters = count_parameters(model)
        criterion = BPRLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    num_parameters += extra_hash_parameter
    mylog.write(f'Number of model parameters: {num_parameters}\n')
    mylog.write_summary({"Number of model parameters": num_parameters})
    if config.train_opt == False: return

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    start_time = time.perf_counter()

    if loss == "BPR":
        train_and_evaluate(
            hash_type,
            model,
            train_loader,
            val_loader,
            test_loader,
            user_clusters,
            item_clusters,
            train_candidates,
            val_candidates,
            test_candidates,
            k,
            criterion,
            optimizer,
            run_name,
            device,
            num_epochs=num_epochs,
            patience=patience,
            mylog=mylog
        )
    else :
        raise NotImplementedError(f'No such loss function: {loss} !')

    torch.cuda.synchronize(device)
    elapsed_time = time.perf_counter() - start_time
    cpu_mem_used = (psutil.Process().memory_info().rss - start_cpu_mem) / 1024 ** 2  # MB
    gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
    mylog.write(f'Cluster_Time: {cluster_cost} s\t Total Time:{elapsed_time} s\t CPU_mem_used:{cpu_mem_used} MB\t GPU_mem_peak:{gpu_mem_peak} MB\n')
    mylog.write_summary({'Total Time (s)': elapsed_time})
    mylog.print_summary()
    mylog.close()


def tensor_size(t):
    def flatten(items):
        for x in items:
            if isinstance(x, (list, tuple)):
                yield from flatten(x)
            else:
                yield x
    t = list(flatten(t))

    return torch.tensor(t).numel()
if __name__ == "__main__":
    main()
