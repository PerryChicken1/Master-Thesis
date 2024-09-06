# CORESET SELECTION BASED ON METADATA DISTRIBUTIONS

import pandas as pd
import numpy as np

class metadata_dist_selector:

    def __init__(self, dataset: pd.DataFrame, coreset_size_target: int=4000, group_column: str='ND_cluster') -> None:
        """
        Coreset selector based on distribution of metadata features. 

        Metadata features are summarised into a `group_column`, and observations are sampled with probability inversely related to their group size.

        Args:
            dataset (pd.DataFrame): dataset including `group_column`
            coreset_size_target (int): desired number of observations in coreset
            group_column (str): name of column determining `group` of observation.
        """
        self.dataset                = dataset
        self.coreset_size_target    = coreset_size_target
        self.group_column           = group_column

        self.coreset_indices    = list()

    @property
    def coreset_(self) -> pd.DataFrame:
        """
        Return current coreset.
        """
        return self.dataset[self.dataset.index.isin(self.coreset_indices)]
    
    @property
    def non_coreset_(self) -> pd.DataFrame:
        """
        Return complement of coreset in dataset.
        """
        return self.dataset[~self.dataset.index.isin(self.coreset_indices)]
    
    @property
    def coreset_size_(self) -> int:
        """
        Return length of current coreset.
        """
        return len(self.coreset_indices)
    
    @staticmethod
    def compute_probabilities(dataset:pd.DataFrame, group_column:str) -> pd.Series:
        """
        Compute sampling probabilities for every observation in dataset.
        Each observation is assigned a probability 1 / (n_k * K), where K = # groups and n_k = size of its group.

        Args:
            dataset (pd.DataFrame): dataset including `group_column`
            group_column (str): name of column determining `group` of observation.

        Returns:
            probabilities (pd.Series): column of sampling probabilities per obsevation.
        """
        # K
        K               = dataset[group_column].nunique()

        # n_k
        group_sizes     = dataset.groupby(group_column).size()
        n_k             = dataset[group_column].map(group_sizes)

        probabilities   = 1 / (n_k * K)

        return probabilities

    @staticmethod
    def select_next_K(non_coreset_: pd.DataFrame, K: int, probability_column: str) -> list:
        """
        Select next K observations with probabilities given in `probability_column`.

        Args:
            non_coreset_ (pd.DataFrame): complement of coreset in dataset
            K (int): number of observations to select
            probability_column (str): name of column determining probability of sampling observation.

        Returns:
            next_K_indices (list): indices in `dataset` of next K observations to add to coreset.
        """
        probabilities   = non_coreset_[probability_column]

        # normalise probabilities
        probabilities   = probabilities / probabilities.sum()

        # select next K
        next_K_indices  = np.random.choice(non_coreset_.index, size=K, replace=False, p=probabilities)

        return next_K_indices.tolist()

    def select_coreset(self, K: int=10) -> list:
        """
        Select coreset.

        Args:
            K (int): number of observations to select in one go before re-computing probabilities.

        Returns:
            coreset_indices (list): indices in `dataset` of observations constituting the coreset.
        """
        # compute initial probabilities
        self.dataset['probabilities']   = self.compute_probabilities(self.dataset, self.group_column)
        
        # until desired coreset size is reached
        while self.coreset_size_ < self.coreset_size_target:

            # select next K obs. to add to coreset
            next_K_indices          = self.select_next_K(self.non_coreset_, K, 'probabilities')
            self.coreset_indices    = self.coreset_indices + next_K_indices
        
        return self.coreset_indices





