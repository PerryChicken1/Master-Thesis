import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cdist
from typing import Self
from collections import deque

class k_center_greedy:
    """
    Custom implementation of the coreset selection method by Sener and Savarese (2018).

    The method chooses a core-set based on the geometry of the dataset.

    Source: https://arxiv.org/abs/1708.00489

    Args:
        dataset (pd.DataFrame): DataFrame of hidden observations
        x (list | str): name(s) of columns containing feature data
        budget (int): size of core-set
        distance_metric (from scipy.spatial.distance): measure of distance between x_i, x_j
    """

    def __init__(self, dataset: pd.DataFrame, x: list | str, budget:float, distance_metric=euclidean):
        
        self.dataset            = dataset
        self.x                  = x
        self.budget             = budget
        self.distance_metric    = distance_metric
        self.coreset_indices    = []
        
        # self.distance_matrix    = np.zeros(shape=(len(dataset), budget))
        self.X_data             = dataset[x].to_numpy()
        self.n_feat             = 1 if type(x) is str else len(x)

    @property
    def coreset_size_(self) -> int:
        """
        Size of coreset.
        """
        return len(self.coreset_indices)
    
    @property
    def coreset_(self) -> pd.DataFrame:
        """
        Coreset.
        """
        return self.dataset[self.dataset.index.isin(self.coreset_indices)]
    
    @property
    def data_pool_(self) -> pd.DataFrame:
        """
        Dataset excluding coreset.
        """
        return self.dataset[~self.dataset.index.isin(self.coreset_indices)]

    def compute_distance_col(self, u_idx: int) -> np.ndarray:
        """
        Compute column of pairwise distances between data and newest center.

        Args:
            u_idx: index of newest center
        """
        # obtain vector of latest center
        center                                  = self.dataset[self.x].loc[u_idx].to_numpy()
        center                                  = center.reshape(1, self.n_feat)

        # compute distances
        distance_col                            = cdist(self.X_data, center, metric=self.distance_metric)
        return distance_col

    def zero_d_coreset(self, distance_col: np.ndarray) -> np.ndarray:
        """
        Replace distances with zeros if point already in coreset.
        """
        # get positional indices of coreset_indices
        np_index                    = self.dataset.index.get_indexer(self.coreset_indices)
        distance_col[np_index]      = 0
        return distance_col

    def update_distances(self, u_idx: int):
        """
        At step t, add column t to distance matrix. Drop an old column if coreset size >= 100. This helps prevent memory issues.

        Args:
            u_idx (int): index of latest center, u
        """
        distance_col                = self.compute_distance_col(u_idx)
        distance_col                = self.zero_d_coreset(distance_col)

        if self.coreset_size_ == 1: 
            self.distance_matrix    = distance_col
        elif self.coreset_size_ > 100: 
            # deque move
            self.distance_matrix    = np.hstack((self.distance_matrix[:, 1:], distance_col))
        else:
            self.distance_matrix    = np.hstack((self.distance_matrix, distance_col))


    def pick_first_center(self) -> int:
        """
        Select first center for core-set.

        Returns:
            u_0: index of first center
        """
        assert self.coreset_size_  == 0, "First center already selected"

        u_0 = np.random.choice(self.dataset.index)

        self.coreset_indices.append(u_0)
        self.update_distances(u_0)

        return u_0

    def pick_next_center(self) -> int:
        """
        Select next center for core-set.

        Returns:
            next_u_idx: index of next center
        """
        # distance from points to their closest centers
        min_distances       = np.min(self.distance_matrix, axis=1)

        # identify point that maximizes the min_distance
        next_u_pos          = np.argmax(min_distances)
        next_u_idx          = self.dataset.index[next_u_pos]

        self.coreset_indices.append(next_u_idx)
        self.update_distances(next_u_idx)
        return next_u_idx

    def pick_center(self) -> int:
        """
        Wrapper for picking the next coreset element.
        """
        if self.coreset_size_ == 0: 
            u = self.pick_first_center()
        else: 
            u = self.pick_next_center()

        return u        

    def run(self) -> Self:
        """
        Construct coreset of size `budget`.
        """
        while self.coreset_size_ < self.budget:
            
            self.pick_center()

            if self.coreset_size_ % 10 == 0:
                
                # print updates every 10 observations
                print(f"{self.coreset_size_} centers added to coreset")
                print(f"Deque size: {self.distance_matrix.shape[-1]}")

        return self
    
