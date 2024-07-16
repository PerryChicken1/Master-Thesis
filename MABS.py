import pandas as pd
import pyogrio
import os
import random
import pandas.api.types as ptypes
import numpy as np
import warnings
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, explained_variance_score, mean_squared_error # , root_mean_squared_error
from matplotlib import pyplot as plt
from scipy.stats import beta
from custom_score_functions import log_ratio
from robust_regression import Torrent

# MAIN BANDIT CLASS

class bandit:
    """
    Bandit to select data for optimizing the model f.

    INPUTS:
    dataset: includes columns x and y
    x: name(s) of independent variable(s)
    y: column name of dependent variable
    features: along which to cluster df, including n_bins if numeric
    T: number of train points to sample before terminating (must be < frac_train * len(dataset))
    batch_size: number of points to sample before computing reward
    frac_train: fraction of dataset for training
    frac_test: fraction of dataset for testing
    frac_val: fraction of dataset for validation
    test_freq: frequency at which to evaluate the model fit on test set
    """

    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict, T: int=1000, batch_size: float=1\
                 , frac_train: float=0.5, frac_test: float=0.48, frac_val: float=0.02, test_freq: int=10):

        # store inputs
        self.dataset            = dataset
        self.x                  = x
        self.y                  = y
        self.features           = features
        self.T                  = T
        self.batch_size         = batch_size
        self.frac_train         = frac_train
        self.frac_test          = frac_test
        self.frac_val           = frac_val
        self.test_freq          = test_freq

        # instantiate lists
        self.hidden_indices     = []
        self.test_indices       = []
        self.val_indices        = []
        self.train_indices      = []
        self.clusters           = []
        self.prev_score         = -np.infty
        self.current_score      = -np.infty
        self.val_scores         = []
        self.test_scores        = []
        self.rewards            = []
        self.sampled_C          = []
        
        # clusters, TTV split, priors
        self.clean_clusters()
        self.generate_clusters(features)
        self.ttv_split()
        self.instantiate_priors()

        # model and score function
        self.model              = PoissonRegressor() # Lasso(tol=1e-2)
        self.predictor_count    = 1 if type(x) is str else len(x)
        self.score              = mean_squared_error # mean_absolute_percentage_error # r2_score
        self.lower_is_better    = True

    def clean_clusters(self):
        """
        Upon initialization, delete all previous cluster assignments.
        """
        cluster_cols = [c for c in self.dataset.columns if 'cluster_ID_' in c]
        self.dataset.drop(columns=cluster_cols, inplace=True)
        self.clusters.clear()

    def generate_cluster(self, feature: str, n_bins: int=10):
        """
        Partitions the data into clusters along the specified metadata variable.
        If the feature is categorical, then the categories determine the clusters.
        If the feature is numeric, then the feature is quantized into n_bins clusters.

        INPUTS:
        feature: name of feature in dataset along which to define clusters.
        n_bins: number of bins, if feature is numeric.

        OUTPUTS:
        Adds column to self.dataset indicating the cluster ID of an observation along the given metadata feature.
        """
        # get column
        try: cluster_column = self.dataset[f'{feature}']
        except KeyError: raise KeyError(f"Column {feature} does not exist in dataset")

        # numeric
        if ptypes.is_numeric_dtype(cluster_column):

            # ?
            cluster_ids, bins   = pd.qcut(cluster_column, q=n_bins, labels=False, duplicates='drop', retbins=True)
            cluster_ids         = cluster_ids.apply(lambda c_id: bins[c_id])

        # categorical
        elif ptypes.is_categorical_dtype(cluster_column):
            
            cluster_ids         = cluster_column

        # unique
        unique_clusters = cluster_ids.unique()

        # save IDs
        feature_name               = f"cluster_ID_{feature}"
        self.dataset[feature_name] = cluster_ids

        # save clusters
        for cluster_name in unique_clusters:
            self.clusters.append((feature_name, cluster_name))
            
    def generate_clusters(self, features: dict):
        """
        Partitions the data into clusters along the specified metadata variables.
        
        INPUTS:
        features: dictionary of form {feature: n_bins}. The n_bins argument is ignored when feature is categorical.
        """
        for feature, n_bins in features.items():
            self.generate_cluster(feature, n_bins)

    def instantiate_priors(self):
        """
        Instantiates prior distributions for Thompson sampling. Beta(1,1)
        """
        self.num_clusters       = len(self.clusters)
        self.alphas             = np.ones(self.num_clusters)
        self.betas              = np.ones(self.num_clusters)
        
    def ttv_split(self):
        """
        Split dataset into train, test and validation components.
        """
        data_size = len(self.dataset.index)
        assert np.isclose(self.frac_train + self.frac_test + self.frac_val, 1.0), "Train, test, validation fractions must sum to 1"

        pos_train           = int(data_size * self.frac_train)
        pos_test            = int(data_size * self.frac_test)

        split_points        = [pos_train, pos_train + pos_test]

        hidden_indices, test_indices, val_indices \
                            = np.split(self.dataset.sample(frac=1), split_points)

        self.hidden_indices = hidden_indices.index.tolist()
        self.test_indices   = test_indices.index.tolist()
        self.val_indices    = val_indices.index.tolist()

    def t_v_shuffle(self):
        """
        Shuffle hidden and validation indices.
        """
        # combine 'n' shuffle
        combined                = self.hidden_indices + self.val_indices
        random.shuffle(combined)

        new_v_indices           = combined[:len(self.hidden_indices)]
        new_h_indices           = combined[len(self.hidden_indices):]

        self.hidden_indices     = new_h_indices
        self.val_indices        = new_v_indices

    def sample_reward_probabilities(self):
        """
        Sample reward probabilities from multivariate Beta distribution.
        """
        pi = np.array([np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]).T
        return pi

    def sample_datapoint(self, pi: np.ndarray):
        """
        Sample a datapoint and add to the training dataset. Return cluster sampled.

        INPUTS:
        pi: reward probabilities
        """
        # find first non-empty cluster from which to sample
        pi_descending   = np.argsort(pi)[::-1]
        counter         = 0

        # runs at least once and until a non-empty cluster is found
        while counter == 0 or cluster.empty:
            j               = pi_descending[counter]
            feature, value  = self.clusters[j]
            
            # implicitly converts index for .isin() to work
            cluster         = self.dataset[(self.dataset[feature] == value)]
            cluster         = cluster[cluster.index.isin(self.hidden_indices)]

            counter += 1
        
        # pick datapoint
        cluster_size    = len(cluster.index)
        s_index         = np.random.randint(0, cluster_size)
        s               = cluster.index[s_index]

        self.train_indices.append(s)
        try: self.hidden_indices.remove(s)
        except ValueError: raise ValueError(f"s = {s}")
        self.sampled_C.append(j)
        return j
    
    def score_prediction(self, y: np.ndarray, y_hat: np.ndarray):
        """
        Compute score for predictions.

        INPUTS:
        y: true values
        y_hat: predicted values
        """
        return self.score(y, y_hat)

    def compute_reward(self):
        """
        Compute reward with current train indices.
        """
        X_train = self.dataset[self.dataset.index.isin(self.train_indices)][self.x].to_numpy()
        y_train = self.dataset[self.dataset.index.isin(self.train_indices)][self.y].to_numpy()
        X_val   = self.dataset[self.dataset.index.isin(self.val_indices)][self.x].to_numpy()
        y_val   = self.dataset[self.dataset.index.isin(self.val_indices)][self.y].to_numpy()

        # reshape if single feature
        if self.predictor_count == 1: 
            X_train = X_train.reshape(-1, 1)
            X_val   = X_val.reshape(-1,1)

        self.X_train= X_train 
        self.model.fit(X_train, y_train)

        y_val_hat           = self.model.predict(X_val)
        
        # negative or near-zero predictions are rounded up
        # y_val_hat           = np.maximum(y_val_hat, 0.001)

        self.prev_score     = self.current_score
        self.current_score  = self.score_prediction(y_val, y_val_hat)

        # allocate reward
        if self.lower_is_better: reward = 1 if self.current_score < self.prev_score else 0
        elif not self.lower_is_better: reward = 1 if self.current_score > self.prev_score else 0

        self.rewards.append(reward)
        self.val_scores.append(self.current_score)

        return reward
    
    def compute_test_score(self):
        """
        Compute score on test set with current train indices.
        """
        X_train = self.dataset[self.dataset.index.isin(self.train_indices)][self.x].to_numpy()
        y_train = self.dataset[self.dataset.index.isin(self.train_indices)][self.y].to_numpy()
        X_test  = self.dataset[self.dataset.index.isin(self.test_indices)][self.x].to_numpy()
        y_test  = self.dataset[self.dataset.index.isin(self.test_indices)][self.y].to_numpy()

        # reshape if single feature
        if self.predictor_count == 1: 
            X_train = X_train.reshape(-1, 1)
            X_test  = X_test.reshape(-1,1)
   
        self.model.fit(X_train, y_train)

        y_test_hat  = self.model.predict(X_test)

        # negative or near-zero predictions are rounded up
        # y_test_hat  = np.maximum(y_test_hat, 0.001)

        test_score  = self.score_prediction(y_test, y_test_hat)
        self.test_scores.append(test_score)

    def update_beta_params(self, reward: int, j_batch: int):
        """
        Update parameters for cluster sampling distributions based on reward.

        INPUTS:
        reward: reward from latest datapoint selection
        j: indices of clusters from which datapoints selected
        """
        for j in j_batch:
            if reward == 1: self.alphas[j] += 1
            else:           self.betas[j] += 1

    def under_the_hood(self, pi:np.ndarray, j:int, current_score: float, prev_score: float, r:float):
        """
        Provide intermittent status reports about the agent during data selection.
        Only valuable when batch_size == 1. Not enabled by default.
         
        INPUTS:
        pi: expected cluster values
        j: index of cluster last sampled
        current_score: score relating to j
        prev_score: previous score
        r: latest reward
        """
        cluster_sampled = self.clusters[j]
        feature         = cluster_sampled[0].replace('cluster_ID_', '')

        print(f"pi = {pi}")
        print(f"Cluster names: {self.clusters}")
        print(f"Cluster j = {j} sampled: {cluster_sampled}")
        print(f"Model score before sample: {prev_score}")
        print(f"Model score after sample: {current_score}")
        print(f"Reward: {r}")

        self.plot_beta_dist(feature=feature)

    def reset(self):
        """
        Clear attributes between algorithm runs.
        """
        self.train_indices.clear()
        self.val_scores.clear()
        self.test_scores.clear()
        self.rewards.clear()
        self.sampled_C.clear()

        self.prev_score         = -np.infty
        self.current_score      = -np.infty
        self.alphas             = np.ones(self.num_clusters)
        self.betas              = np.ones(self.num_clusters)

        self.ttv_split()

    def run_full_model(self):
        """
        Fit model to full hidden data to benchmark MABS.
        """
        # hidden data becomes train data
        self.train_indices      = self.hidden_indices[:]

        # compute test score
        self.compute_test_score()

        # pad test scores for plotting
        n_tests                 = (self.T // self.test_freq)
        self.test_scores        = self.test_scores * n_tests
    
    def run_random_baseline(self):
        """
        Run a random datapoint selector to benchmark MABS.
        """
        t = 1

        # run for T time steps
        while t <= self.T:
            
            s   = np.random.choice(self.hidden_indices)
            self.train_indices.append(s)
            self.hidden_indices.remove(s)

            _   = self.compute_reward()
            
            if t % self.test_freq == 0: self.compute_test_score()
            
            t += 1

    def run_TORRENT(self):
        """
        Run TORRENT method to benchmark MABS.
        """
        # find T 'inliers'
        a                       = self.T / len(self.dataset)

        self.train_indices      = self.hidden_indices[:]

        # train / test split
        X_train_tor             = self.dataset[self.dataset.index.isin(self.train_indices)][self.x].to_numpy()
        y_train_tor             = self.dataset[self.dataset.index.isin(self.train_indices)][self.y].to_numpy()
        X_test_tor              = self.dataset[self.dataset.index.isin(self.test_indices)][self.x].to_numpy()
        y_test_tor              = self.dataset[self.dataset.index.isin(self.test_indices)][self.y].to_numpy()

        # fit torrent
        torrent                 = Torrent(a)
        torrent.fit(X_train_tor, y_train_tor)

        # inliers
        X_inliers_tor           = X_train_tor[torrent.inliers]
        y_inliers_tor           = y_train_tor[torrent.inliers]

        # score model fit on inliers
        self.model.fit(X_inliers_tor, y_inliers_tor)
        y_pred_tor              = self.model.predict(X_test_tor)
        torrent_score           = self.score_prediction(y_test_tor, y_pred_tor)

        # store score
        n_tests                 = (self.T // self.test_freq)
        self.test_scores        = [torrent_score] * n_tests

    def run_MABS(self):
        """
        Run the multi-armed bandit selection algorithm.
        """
        t = 1

        # run for T time steps
        while t <= self.T:

            j_batch = []

            # run for batch_size obs before computing reward
            for _ in range(self.batch_size):
                pi  = self.sample_reward_probabilities()
                j   = self.sample_datapoint(pi)
                j_batch.append(j)

            r = self.compute_reward()

            if t % self.test_freq == 0: 
                self.compute_test_score()
                self.t_v_shuffle()
                
            self.update_beta_params(reward=r, j_batch=j_batch)

            # self.under_the_hood(pi, j, self.current_score, self.prev_score, r) #TODO: remove

            t += 1

    def plot_scores(self, times:np.ndarray, scores: np.ndarray, *args, **kwargs):
        """
        After an algorithm is run, plot the scores over time.

        INPUTS:
        times: list of times
        scores: list of scores
        *args and **kwargs: parameters for plt.plot()
        """
    
        plt.plot(times, scores, markersize=2, *args, **kwargs)
        plt.xlabel("Time step t")
        plt.ylabel(f"{self.score.__name__}")
        plt.title("Regression model performance over time")

    def plot_beta_dist(self, feature: str, *args, **kwargs):
        """
        After run_MABS() is executed, plot the beta distributions for all categories of a given feature. 

        INPUTS:
        *args and **kwargs: parameters for plt.plot()
        """
        # filter to relevant features & categories
        feature_name    = f"cluster_ID_{feature}"
        j_indices       = [index for index, (name, _) in enumerate(self.clusters) if name == feature_name]
        categories      = [cat for (name, cat) in self.clusters if name == feature_name]

        # get relevant alphas & betas
        alphas  = self.alphas[j_indices]
        betas   = self.betas[j_indices]

        # plot         
        x = np.arange(100) / 100
        plt.clf()

        for k, _ in enumerate(j_indices):
            y = beta.pdf(x=x, a=alphas[k], b=betas[k])
            plt.plot(x, y, ls='-', linewidth=2, label=categories[k], *args, **kwargs)

        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title(f"Sampling distributions at t = {self.T}")
        plt.suptitle(f"Feature {feature}")
        plt.legend()
        plt.show()

    def eval_test_performance(self, n_runs:int = 1, which: str = "MABS"):
        """
        Compute average test scores over n_runs for MABS or one of the baselines.

        INPUTS:
        n_runs: number of times to repeat the algorithms
        which: whether to evaluate 'MABS', 'rb', 'TORRENT' or 'full'
        
        OUTPUTS:
        avg_scores: score at every test time (np.ndarray)
        """
        test_scores     = np.zeros((self.T // self.test_freq))
        current_run     = 1

        while current_run <= n_runs:
            
            print(f"Benchmarking run {current_run} for model {which}")

            self.reset()
            if which == 'MABS': self.run_MABS()
            elif which == 'rb': self.run_random_baseline()
            elif which == 'full': self.run_full_model()
            elif which == 'TORRENT': self.run_TORRENT()

            test_scores         = np.sum((test_scores , self.test_scores), axis=0)

            current_run += 1

        avg_scores  = test_scores / n_runs

        return avg_scores
    
    def plot_test_performance(self, avg_scores_dict: dict, ylim: tuple):
        """
        Plot outputs from eval_test_performance().

        INPUTS:
        avg_scores_dict: {label: avg_scores}
        ylim: (y_min, y_max)
        """
        plt.figure()
        times   = [t for t in range(0, self.T + 1)]

        for label, avg_scores in avg_scores_dict.items():
            self.plot_scores(times=times[self.test_freq::self.test_freq], scores=avg_scores, label=label)

        plt.ylim(ylim)
        plt.legend()
        plt.show()
    
    def benchmark_MABS(self, n_runs:int = 1):
        """
        Plot average test scores over n_runs for random selector, full model and MABS.

        INPUTS:
        n_runs: number of times to repeat the algorithms
        """
        # random baseline, MABS with all features and full model
        avg_scores_full         = self.eval_test_performance(n_runs, "full")
        avg_scores_rb           = self.eval_test_performance(n_runs, "rb")
        avg_scores_TORRENT      = self.eval_test_performance(n_runs, "TORRENT")
        avg_scores_MABS         = self.eval_test_performance(n_runs, "MABS")
        avg_scores_dict         = {'Random baseline': avg_scores_rb, 'MABS': avg_scores_MABS\
                                   , 'Full model': avg_scores_full, 'TORRENT': avg_scores_TORRENT}

        # each individual feature
        if len(self.features) > 1:

            for feature, n_bins in self.features.items():

                continue # TODO

                # generate cluster
                self.clean_clusters()
                self.generate_cluster(feature, n_bins)

                avg_scores_f    = self.eval_test_performance(n_runs, "MABS")

                avg_scores_dict[f"MABS {feature}"] \
                                =  avg_scores_f

        y_min                   = min([np.min(value) for _, value in avg_scores_dict.items()])
        y_max                   = max([np.max(value) for _, value in avg_scores_dict.items()])
        
        # y_max                   = min([np.max(value) for _, value in avg_scores_dict.items()]) #TODO
        
        self.plot_test_performance(avg_scores_dict, (y_min, y_max))    

# BANDIT ADVERSARIALLY ATTACKS TRAINING DATA

class crafty_bandit(bandit):
    """
    Corrupts training data and attempts to successfully model regardless.

    INPUTS:
    dataset: includes columns x and y
    x: name(s) of independent variable(s)
    y: column name of dependent variable
    features: along which to cluster df, including n_bins if numeric
    T: number of train points to sample before terminating (must be < frac_train * len(dataset))
    batch_size: number of points to sample before computing reward
    frac_train: fraction of dataset for training
    frac_test: fraction of dataset for testing
    frac_val: fraction of dataset for validation
    test_freq: frequency at which to evaluate the model fit on test set
    p_corrupt: proportion of training observations to corrupt
    """
    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict, T: int=1000, batch_size: float=1\
                 , frac_train: float=0.5, frac_test: float=0.48, frac_val: float=0.02, test_freq: int=10, p_corrupt: float=0.1):

        super().__init__(dataset=dataset, x=x, y=y, features=features, T=T, batch_size=batch_size\
                 , frac_train=frac_train, frac_test=frac_test, frac_val=frac_val, test_freq=test_freq)

        # clusters and TTV split
        self.ttv_split()
        self.corrupt_train_data(p_corrupt)
        self.clean_clusters()
        self.generate_clusters(self.features)

    def corrupt_train_data(self, p_corrupt: float):
        """
        Corrupt training datapoints Y and record.

        INPUTS:
        p_corrupt: proportion of training observations to corrupt
        """
        # corrupt training points
        self.dataset["is_corrupted"]    = 0
        min_y, max_y                    = min(self.dataset[self.y]), max(self.dataset[self.y])

        for i in self.hidden_indices:
            u = np.random.uniform(0,1)

            if u < p_corrupt:
                random_y                            = np.random.randint(min_y, max_y) # self.dataset[self.y].sample(1)
                self.dataset.at[i, self.y]          = random_y
                self.dataset.at[i, "is_corrupted"]  = 1
        
        # make column categorical; add feature to dataset
        self.dataset["is_corrupted"]    = self.dataset["is_corrupted"].astype('category')
        self.features['is_corrupted']   = None

# LAZY BANDIT DOES NOT PERFORM TTV SPLIT

class lazy_bandit(bandit):
    """
    Like bandit, but too lazy to shuffle the data. 
    User specifies hidden (train), validation and test indices.

    INPUTS:
    dataset: includes columns x and y
    x: name(s) of independent variable(s)
    y: column name of dependent variable
    features: along which to cluster df, including n_bins if numeric
    T: number of train points to sample before terminating (must be < frac_train * len(dataset))
    batch_size: number of points to sample before computing reward
    hidden_indices: indices of datapoints eligible for training
    test_indices: indices of datapoints for testing
    val_indices: indices of datapoints for validation
    test_freq: frequency at which to evaluate the model fit on test set
    """
    
    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict\
                 , hidden_indices: list, test_indices: list, val_indices: list\
                 , T: int=1000, batch_size: float=1, test_freq: int=10):

        super().__init__(dataset, x, y, features, T, batch_size, 0, 0, 0, test_freq) # TODO: replace 0 -> 1 if issues
        
        # copy to avoid mutation issues
        self.init_hidden_indices= hidden_indices[:]
        self.hidden_indices     = hidden_indices
        self.test_indices       = test_indices
        self.val_indices        = val_indices
        
    def ttv_split(self):
        """
        Overwrite ttv_split() to do nothing
        """
        pass

    def reset(self):
        """
        Restore initial hidden indices with reset()
        """
        super().reset()

        # restore hidden indices
        self.hidden_indices     = self.init_hidden_indices[:]

# WHETHER TO SHUFFLE DATA OR ACCEPT TTV SPLIT (NOT USED)

def bandit_shuffler(shuffle:bool = True):
    """
    Whether to shuffle data for sub-bandit or accept a TTV split.
    Use case is ND_bandit.

    INPUTS:
    shuffle: True or False

    OUTPUTS:
    instance of class with shuffle implemented (or not)
    """

    def class_shuffler(cls):
        """
        INPUTS:
        cls: class

        OUTPUTS:
        class with desired parent
        """

        parent = bandit if shuffle else lazy_bandit

        class shuffle_child(parent):
            pass
            
        return shuffle_child
    
    return class_shuffler

# ND BANDIT PERMITS N-DIMENSIONAL CLASSES

class ND_bandit(bandit):
    """
    Bandit permitting n-dimensional classes (i.e., comprising n features).
    All features are crossed to generate clusters.

    INPUTS:
    dataset: includes columns x and y
    x: name(s) of independent variable(s)
    y: column name of dependent variable
    features: along which to cluster df, including n_bins if numeric
    T: number of train points to sample before terminating (must be < frac_train * len(dataset))
    batch_size: number of points to sample before computing reward
    frac_train: fraction of dataset for training
    frac_test: fraction of dataset for testing
    frac_val: fraction of dataset for validation
    test_freq: frequency at which to evaluate the model fit on test set
    """
    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict\
                 , T: int=1000, batch_size: float=1, frac_train: float=0.5, frac_test: float=0.48\
                 , frac_val: float=0.02, test_freq: int=10):
  
        # instantiate for mapping clusters to features
        self.cluster_dict       = dict()

        super().__init__(dataset=dataset, x=x, y=y, features=features, T=T, batch_size=batch_size, frac_train=frac_train\
                         , frac_test=frac_test, frac_val=frac_val, test_freq=test_freq)

    def generate_clusters(self, features: dict):
        """
        Generate a cluster for every entry in the cartesian product of binned features.

        INPUTS:
        features: {feature: n_bins} dictionary

        OUTPUTS:
        Column 'ND_cluster' added to self.dataset indicating cluster membership.
        """
        # generate unit clusters
        super().generate_clusters(features)

        # unique combinations
        dataset_clusters                        = self.dataset.filter(like='cluster_ID_')
        unique_combinations                     = dataset_clusters.drop_duplicates().reset_index(drop=True)

        # map cluster to feature values
        for idx, row in unique_combinations.iterrows():

            row_dict                            = row.to_dict()
            self.cluster_dict[idx]              = row_dict
        
        # 1D -> ND clusters
        unique_combinations['ND_cluster']       = unique_combinations.index
        dataset_merged                          = pd.merge(self.dataset, unique_combinations, on=list(dataset_clusters.columns), how='left')
        dataset_merged.drop(dataset_clusters.columns, axis=1, inplace=True)

        # delete unit clusters
        self.clean_clusters()

        # retain ND clusters
        self.clusters                           = [('ND_cluster', value) for value in unique_combinations['ND_cluster'].unique()]
        self.dataset                            = dataset_merged

    def under_the_hood(self, pi:np.ndarray, j:int, current_score: float, prev_score: float, r:float):
        """
        Provide intermittent status reports about the agent during data selection.
        Only valuable when batch_size == 1. Not enabled by default.
        plot_beta_dist() call suppressed: too much customizability.
         
        INPUTS:
        pi: expected cluster values
        j: index of cluster last sampled
        current_score: score relating to j
        prev_score: previous score
        r: latest reward
        """
        cluster_sampled = self.cluster_dict[j]

        print(f"pi = {pi}")
        print(f"Cluster j = {j} sampled: {cluster_sampled}")
        print(f"Model score before sample: {prev_score}")
        print(f"Model score after sample: {current_score}")
        print(f"Reward: {r}")

        
    def plot_beta_dist(self, feature: str, fixed_val,*args, **kwargs):
        """
        After run_MABS() is executed, plot the beta distributions at a fixed value of a feature.

        INPUTS:
        feature: to fix
        fixed_val: fixed value of feature
        *args and **kwargs: parameters for plt.plot()
        """
        # filter where feature == fixed_val
        indices         = [idx for idx, feat_dict in self.cluster_dict.items() if feat_dict["cluster_ID_" + feature] == fixed_val]
        
        # concatenate labels
        label_pieces    = [f"{feature} = {value}" for idx in indices for feature, value in self.cluster_dict[idx].items()]
        feat_count      = len(self.features)
        labels          = [', '.join(label_pieces[i:i+feat_count]) for i in range(0, len(label_pieces), feat_count)] 

        # get relevant alphas & betas
        alphas          = self.alphas[indices]
        betas           = self.betas[indices]

        # plot         
        x               = np.arange(100) / 100
        plt.clf()

        for k, _ in enumerate(indices):
            y           = beta.pdf(x=x, a=alphas[k], b=betas[k])
            plt.plot(x, y, ls='-', linewidth=2, label=labels[k], *args, **kwargs)

        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title(f"Sampling distributions at t = {self.T}")
        plt.suptitle(f"Feature {feature} = {fixed_val}")
        plt.legend(fontsize='small')
        plt.show()

