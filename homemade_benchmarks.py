import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cdist
from typing import Self

class k_center_greedy:
    """
    Implementation of the coreset selection method by Sener and Savarese (2018).

    The method chooses a core-set based on the geometry of the dataset.

    Source: https://arxiv.org/abs/1708.00489

    Args:
        dataset_train: DataFrame of TRAIN observations
        x: name(s) of columns containing feature data
        budget: size of core-set
        distance_metric: measure of distance between x_i, x_j
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

    def update_distances(self, u_idx: int):
        """
        At step t, add column t to distance matrix.

        Args:
            u_idx: index of latest center, u
        """
        distance_col                            = self.compute_distance_col(u_idx)
        self.distance_matrix                    = np.hstack((self.distance_matrix, distance_col))

    def pick_first_center(self) -> int:
        """
        Select first center for core-set. 
        """
        assert self.coreset_size_  == 0, "First center already selected"

        u_0                     = np.random.choice(self.dataset.index)
        self.coreset_indices.append(u_0)
        self.distance_matrix    = self.compute_distance_col(u_0)
        return u_0

    def pick_next_center(self) -> int:
        """
        Select next center for core-set.
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
        Call appropriate function.
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

                print(f"{self.coreset_size_} centers added to coreset")

        return self