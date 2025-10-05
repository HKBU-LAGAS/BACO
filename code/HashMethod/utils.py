import numpy as np
import scipy.sparse as sp

def map_to_consecutive_integers_multi(clusters):
    """将多维标签列表中的值映射到从0开始的连续整数空间.

        Args:
            clusters: list of lists, 例如 [[5,3], [5,8], [3]]

        Returns:
            mapped_clusters: 映射后的标签集合，例如 [[1,0], [1,2], [0]]
            num_unique: 映射后的唯一元素数量
        """
    # 展平所有标签并提取唯一值（按排序顺序）
    all_labels = np.concatenate(clusters) if isinstance(clusters, list) else clusters.flatten()
    label_count = len(all_labels)
    unique_labels = np.unique(all_labels)
    # 创建映射字典：旧标签 -> 新连续ID
    label_to_id = {old: new for new, old in enumerate(unique_labels)}
    # 逐元素替换标签
    mapped_clusters = [
        [label_to_id[label] for label in sublist]
        for sublist in clusters
    ]
    return mapped_clusters, len(unique_labels), label_count

def map_to_consecutive_integers(values):
    """
    Maps each unique value in the input list to a unique consecutive integer.

    Args:
    values (list): The list of values to be mapped.

    Returns:
    tuple: A tuple containing the mapped list and the dictionary of value to integer mapping.
    """
    # Get unique values and sort them
    unique_values = sorted(set(values))

    # Create a mapping from unique values to consecutive integers
    value_to_int = {val: idx for idx, val in enumerate(unique_values)}

    # Map original values to the consecutive integers
    mapped_values = [value_to_int[val] for val in values]

    return np.array(mapped_values), len(unique_values)

def double_map_to_consecutive_integers(values1, values2):
    """
    Maps each unique value in the input list to a unique consecutive integer.

    Args:
    values (list): The list of values to be mapped.

    Returns:
    tuple: A tuple containing the mapped list and the dictionary of value to integer mapping.
    """
    # Get unique values and sort them
    unique_values = sorted(set(np.concatenate([values1, values2])))

    # Create a mapping from unique values to consecutive integers
    value_to_int = {val: idx for idx, val in enumerate(unique_values)}

    # Map original values to the consecutive integers
    mapped_values1 = [value_to_int[val] for val in values1]
    mapped_values2 = [value_to_int[val] for val in values2]

    return np.array(mapped_values1), np.array(mapped_values2), len(unique_values)

def build_block(mat):
    """
        Construct a symmetric block matrix from a given matrix.

        This function creates a larger block matrix of shape (n_user+n_item, n_user+n_item)
        from the input matrix `mat` by placing:
        1. The original matrix in the top-right quadrant.
        2. The transpose of the original matrix in the bottom-left quadrant.
        3. Zeros in the top-left and bottom-right quadrants.

        Parameters:
        mat (scipy.sparse.spmatrix): Input matrix in any sparse format (will be converted to COO).

        Returns:
        scipy.sparse.csr_matrix: Symmetric block matrix in CSR format.

        Shape:
        - Input matrix shape: (n_user, n_item)
        - Output matrix shape: (n_user + n_item, n_user + n_item)

        Example:
        Input matrix `mat`:
            [[1, 2],
             [3, 4]]
        Output block matrix:
            [[0, 0, 1, 2],
             [0, 0, 3, 4],
             [1, 3, 0, 0],
             [2, 4, 0, 0]]
        """
    mat_coo = mat.tocoo()
    n_user, n_item = mat.shape
    # Upper-right block coordinates
    upper_rows = mat_coo.row
    upper_cols = mat_coo.col + n_user
    # Lower-left block coordinates (transposed)
    lower_rows = mat_coo.col + n_user
    lower_cols = mat_coo.row
    # Combine coordinates and data
    all_rows = np.concatenate([upper_rows, lower_rows])
    all_cols = np.concatenate([upper_cols, lower_cols])
    all_data = np.concatenate([mat_coo.data, mat_coo.data])
    return sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(n_user+n_item, n_user+n_item)).tocsr()
