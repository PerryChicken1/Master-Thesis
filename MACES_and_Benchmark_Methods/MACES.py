import pandas as pd
import random
import pandas.api.types as ptypes
import numpy as np
import pickle as pkl
import torch
import warnings
import functools
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, explained_variance_score, mean_squared_error # , root_mean_squared_error
from matplotlib import pyplot as plt
from scipy.stats import beta
from scipy.spatial.distance import euclidean
from Models_and_Helper_Functions.custom_score_functions import log_ratio
from MACES_and_Benchmark_Methods.robust_regression import Torrent
from Models_and_Helper_Functions.custom_models import *
from MACES_and_Benchmark_Methods.ll_coreset_selector import LL_coreset_selector
from MACES_and_Benchmark_Methods.k_center_greedy import k_center_greedy
from MACES_and_Benchmark_Methods.metadata_dists import metadata_dist_selector

# MAIN BANDIT CLASS

class bandit:
    """
    Bandit to select coreset for optimizing the model f according to MACES method (Perry, 2024).

    Performs hidden-test-validation split on the observation level according to `frac_` inputs.
    """

    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict, T: int=1000, batch_size: int=10\
                 , frac_train: float=0.5, frac_test: float=0.48, frac_val: float=0.02, test_freq: int=5, model=MLP()):
        
        """   
        Args:
            dataset: includes columns x and y
            x: name(s) of independent variable(s)
            y: column name of dependent variable
            features: along which to cluster df, including n_bins if numeric
            T: number of train points to sample before terminating (must be < frac_train * len(dataset))
            batch_size: number of points to sample before computing reward
            frac_train: fraction of dataset for training
            frac_test: fraction of dataset for testing
            frac_val: fraction of dataset for validation
            test_freq: frequency (in batches) at which to evaluate the model fit on test set
            model: object with fit() and predict() methods
        """

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
        self.test_freq_t        = test_freq * batch_size

        # instantiate lists
        self.clusters           = []
        self.prev_score         = -np.infty
        self.current_score      = -np.infty
        self.val_scores         = []
        self.test_scores        = []
        self.rewards            = []
        self.sampled_C          = []
        self.test_times         = range(self.test_freq_t, T+1, self.test_freq_t)
        
        # clusters, TTV split, priors
        self.ttv_split()
        self.clean_clusters()
        self.generate_clusters(features)
        self.instantiate_priors()

        # model and score function
        self.model              = model
        self.score              = mean_squared_error
        self.lower_is_better    = True

        # identifier for `self.store_results()`
        self.results_idx        = 0
        self.results_dict       = dict()

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

        Args:
            feature: name of feature in dataset along which to define clusters.
            n_bins: number of bins, if feature is numeric.

        Returns:
            Adds column to self.dataset indicating the cluster ID of an observation along the given metadata feature.
        """
        # get column
        try: cluster_column = self.dataset[f'{feature}']
        except KeyError: raise KeyError(f"Column {feature} does not exist in dataset")

        # numeric
        if ptypes.is_numeric_dtype(cluster_column):

            # use only hidden observations to determine clusters
            cluster_col_hidden  = cluster_column[self.dataset.index.isin(self.hidden_indices)]
            _, bins_h           = pd.qcut(cluster_col_hidden, q=n_bins, labels=False, duplicates='drop', retbins=True)
            bins_h              = np.sort(bins_h)
            cluster_ids         = pd.cut(cluster_column, bins=bins_h, labels=False, include_lowest=True)

            # continuous
            # cluster_ids, bins   = pd.qcut(cluster_column, q=n_bins, labels=False, duplicates='drop', retbins=True)
            # cluster_ids         = cluster_ids.apply(lambda c_id: bins[c_id])

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
        
        Args:
            features: dictionary of form {feature: n_bins}. The n_bins argument is ignored when feature is categorical.
        """
        for feature, n_bins in features.items():
            self.generate_cluster(feature, n_bins)

    @property
    def num_clusters_(self):
        """
        Number of clusters (read-only property)
        """
        return len(self.clusters)
    
    @property
    def predictor_count_(self):
        """
        Number of predictors in x (read-only property)
        """
        if type(self.x) is str: return 1
        else: return len(self.x)

    def dataset_splitter(self, parts: list = ['hidden', 'train', 'test', 'validation']) -> np.ndarray:
        """
        Split dataset into hidden, train, test and validation parts. X before y, H -> Tr -> Te -> V.

        Args:
            parts: sub-list of ['hidden', 'train', 'test', 'validation'] specifying which arrays to return, in that order.

        Returns:
            return_tup: sub-tuple of (X_hidden, y_hidden, X_train, y_train, X_test, y_test, X_val, y_val)
        """
        assert all(part in ['hidden', 'train', 'test', 'validation'] for part in parts),\
            "Invalid dataset part detected!"

        return_tup = tuple()

        if 'hidden' in parts:
            X_hidden    = self.dataset[self.dataset.index.isin(self.hidden_indices)][self.x].to_numpy(dtype='float32')
            y_hidden    = self.dataset[self.dataset.index.isin(self.hidden_indices)][self.y].to_numpy(dtype='float32')
            return_tup  = (*return_tup, X_hidden, y_hidden)

        if 'train' in parts:
            X_train     = self.dataset[self.dataset.index.isin(self.train_indices)][self.x].to_numpy(dtype='float32')
            y_train     = self.dataset[self.dataset.index.isin(self.train_indices)][self.y].to_numpy(dtype='float32')
            return_tup  = (*return_tup, X_train, y_train)
        
        if 'test' in parts:
            X_test      = self.dataset[self.dataset.index.isin(self.test_indices)][self.x].to_numpy(dtype='float32')
            y_test      = self.dataset[self.dataset.index.isin(self.test_indices)][self.y].to_numpy(dtype='float32')
            return_tup  = (*return_tup, X_test, y_test)

        if 'validation' in parts:
            X_val       = self.dataset[self.dataset.index.isin(self.val_indices)][self.x].to_numpy(dtype='float32')
            y_val       = self.dataset[self.dataset.index.isin(self.val_indices)][self.y].to_numpy(dtype='float32')
            return_tup  = (*return_tup, X_val, y_val)

        return return_tup

    def instantiate_priors(self):
        """
        Instantiates prior distributions for Thompson sampling. Beta(1,1)
        """
        self.alphas             = np.ones(self.num_clusters_)
        self.betas              = np.ones(self.num_clusters_)
        
    def ttv_split(self):
        """
        Split dataset into train, test and validation components.
        """
        data_size           = len(self.dataset.index)
        assert np.isclose(self.frac_train + self.frac_test + self.frac_val, 1.0), "Train, test, validation fractions must sum to 1"

        pos_train           = int(data_size * self.frac_train)
        pos_test            = int(data_size * self.frac_test)

        split_points        = [pos_train, pos_train + pos_test]

        hidden_indices, test_indices, val_indices \
                            = np.split(self.dataset.sample(frac=1), split_points)

        self.hidden_indices = hidden_indices.index.tolist()
        self.test_indices   = test_indices.index.tolist()
        self.val_indices    = val_indices.index.tolist()
        self.train_indices  = []

    def t_v_shuffle(self):
        """
        Shuffle hidden and validation indices. Idea is to produce a more robust fit.
        """
        # combine 'n' shuffle
        combined                = self.hidden_indices + self.val_indices
        random.shuffle(combined)

        new_h_indices           = combined[:len(self.hidden_indices)]
        new_v_indices           = combined[len(self.hidden_indices):]

        self.hidden_indices     = new_h_indices
        self.val_indices        = new_v_indices

    def sample_reward_probabilities(self):
        """
        Sample reward probabilities from multivariate Beta distribution.
        """
        pi = np.array([np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]).T
        return pi

    def sample_batch(self, pi: np.ndarray, batch_size:int):
        """
        Sample batch_size datapoints and add them to the training dataset. Return cluster sampled.

        Args:
            pi: reward probabilities
            batch_size: number of datapoints to sample

        Returns:
            j: cluster from which sample is taken
            sample_size: size of sample taken
        """
        # find first non-empty cluster from which to sample
        pi_descending   = np.argsort(pi)[::-1]
        counter         = 0
        cluster_size    = 0

        # runs at least once and until a cluster of size batch_size is found
        while cluster_size < batch_size:
            j               = pi_descending[counter]
            feature, value  = self.clusters[j]
            
            # implicitly converts index for .isin() to work
            cluster         = self.dataset[(self.dataset[feature] == value)]
            cluster         = cluster[cluster.index.isin(self.hidden_indices)]

            cluster_size    = len(cluster.index)
            counter         +=1

        # train indices
        s_indices       = np.random.choice(a=cluster_size, size=batch_size, replace=False)
        
        # store indices
        for idx in s_indices: 
            s           = cluster.index[idx]
            self.train_indices.append(s)

            try: self.hidden_indices.remove(s)
            except ValueError: raise ValueError(f"s = {s}")
        
        self.sampled_C.append(j)
        return j
    
    def score_prediction(self, y: np.ndarray, y_hat: np.ndarray):
        """
        Compute score for predictions.

        Args:
            y: true values
            y_hat: predicted values
        """
        return self.score(y, y_hat)

    def compute_reward(self):
        """
        Compute reward with current train indices.
        """
        X_train, y_train, X_val, y_val \
                            = self.dataset_splitter(['train', 'validation'])

        # reshape if single feature
        if self.predictor_count_ == 1: 
            X_train         = X_train.reshape(-1, 1)
            X_val           = X_val.reshape(-1,1)

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
        X_train, y_train, X_test, y_test \
                    = self.dataset_splitter(['train', 'test'])

        # reshape if single feature
        if self.predictor_count_ == 1: 
            X_train = X_train.reshape(-1, 1)
            X_test  = X_test.reshape(-1,1)

        # (REMOVED) double epoch number for evaluation
        # if isinstance(self.model, MLP): self.model.num_epochs *= 2
        self.model.fit(X_train, y_train)
        # if isinstance(self.model, MLP): self.model.num_epochs //= 2

        y_test_hat  = self.model.predict(X_test)

        test_score  = self.score_prediction(y_test, y_test_hat)
        self.test_scores.append(test_score)

    def update_beta_params(self, reward: int, j: int):
        """
        Update parameters for cluster sampling distributions based on reward.

        Args:
            reward: reward from latest datapoint selection
            j: index of cluster from which datapoints selected
        """
        if reward == 1: self.alphas[j] += 1
        else:           self.betas[j] += 1

    def under_the_hood(self, pi:np.ndarray, j:int, current_score: float, prev_score: float, r:float):
        """
        Provide intermittent status reports about the agent during data selection.
        Only valuable when batch_size == 1. Not enabled by default.
         
        Args:
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
        self.instantiate_priors()

        # reinit model, if possible
        try: self.model.reinit_model()
        except AttributeError: pass # print("Model reset not possible")

        self.ttv_split()

    def run_full_model(self):
        """
        Fit model to full hidden data to benchmark MACES.
        """
        # hidden data becomes train data
        self.train_indices      = self.hidden_indices[:]

        # compute test score
        self.compute_test_score()

        # pad test scores for plotting
        n_tests                 = len(self.test_times)
        self.test_scores        = self.test_scores * n_tests
    
    def run_random_baseline(self):
        """
        Run a random datapoint selector to benchmark MACES.
        """
        t = 0

        # run for T time steps
        while t < self.T:
            
            # select random datapoint
            s   = np.random.choice(self.hidden_indices)
            self.train_indices.append(s)
            self.hidden_indices.remove(s)

            # increment
            t   += 1

            # test score
            if t % self.test_freq_t == 0: 
                self.compute_test_score()

    def run_TORRENT(self):
        """
        Run TORRENT method to benchmark MACES.
        """
        # find T 'inliers'
        a                       = self.T / len(self.hidden_indices)

        # train / test split
        X_hidden_tor, y_hidden_tor, X_test_tor, y_test_tor \
                                = self.dataset_splitter(['hidden', 'test'])
        
        # fit torrent
        torrent                 = Torrent(a)
        torrent.fit(X_hidden_tor, y_hidden_tor)

        # inliers
        X_inliers_tor           = X_hidden_tor[torrent.inliers]
        y_inliers_tor           = y_hidden_tor[torrent.inliers]

        # score model fit on inliers
        self.model.fit(X_inliers_tor, y_inliers_tor)
        y_pred_tor              = self.model.predict(X_test_tor)
        torrent_score           = self.score_prediction(y_test_tor, y_pred_tor)

        # store score
        n_tests                 = len(self.test_times)
        self.test_scores        = [torrent_score] * n_tests

        # store train_indices
        train_indices           = [self.hidden_indices[i] for i in torrent.inliers_]
        self.train_indices      = train_indices

        # store torrent to see iter_count & # inliers. TODO: will remove later
        self.torrent            = torrent
    
    def run_k_centers_greedy(self):
        """
        Run the greedy k centers algorithm.
        """
        # instantiate k center agent
        hidden_data         = self.dataset.loc[self.hidden_indices]

        # apply KCG on random subset of data (faster)
        hidden_subset       = hidden_data.sample(frac=0.1, ignore_index=False)

        k_center_agent      = k_center_greedy(hidden_subset, self.x, self.T, euclidean)    

        # pick core-set
        while k_center_agent.coreset_size_ <  k_center_agent.budget:
            
            u               = k_center_agent.pick_center()
            self.train_indices.append(u)
            self.hidden_indices.remove(u)

            if k_center_agent.coreset_size_ % self.test_freq_t == 0:
                self.compute_test_score()

    def run_LL(self):
        """
        Run the loss learning method.
        """
        # lr are left as defaults
        selector  = LL_coreset_selector(n_predictors=self.model.n_predictors, 
                                        num_epochs=self.model.num_epochs, 
                                        print_freq=self.model.print_freq, 
                                        with_scheduler=self.model.with_scheduler, 
                                        loss_fn=self.model.loss_fn,
                                        FC_dim=32)

        # select `self.batch_size` random datapoints to begin
        init_indices    = random.sample(self.hidden_indices, self.batch_size)

        for idx in init_indices:
            self.train_indices.append(idx)
            self.hidden_indices.remove(idx)

        t               = self.batch_size

        while t < self.T:
            
            # fit model to current coreset
            X_hidden, _, X_train, y_train\
                        = self.dataset_splitter(['hidden', 'train'])
            selector    = selector.fit(X_train, y_train)

            # estimate model loss on every hidden observation
            losses      = selector.predict_losses(X_hidden)

            # retrieve indices of hidden obs in descending order of loss
            top_K_idx   = selector.top_K_losses(losses, K=self.batch_size, indices=self.hidden_indices)

            # add the `batch_size` worst observations to the coreset
            for idx in top_K_idx:
                self.train_indices.append(idx)
                self.hidden_indices.remove(idx)

            t           += self.batch_size

            if t % self.test_freq_t == 0:
                self.compute_test_score()
    
    def run_metadist_selector(self):
        """
        Run meta-data distribution selector.
        """
        t                               = 0
        group_column                    = 'ND_cluster'
        K                               = self.batch_size

        assert group_column in self.dataset.columns, f"Metadist selector requires that {group_column} be present in dataset"

        # initialise
        hidden_dataset                  = self.dataset[self.dataset.index.isin(self.hidden_indices)]
        mds                             = metadata_dist_selector(hidden_dataset, self.T, group_column)
        mds.dataset['probabilities']    = mds.compute_probabilities(mds.dataset, group_column)

        # collect T observations
        while t < self.T:

            next_K_indices              = mds.select_next_K(mds.non_coreset_, K, 'probabilities', update_indices=True)

            for idx in next_K_indices:
                self.hidden_indices.remove(idx)
                self.train_indices.append(idx)
            
            t                           +=  K

            # test score
            if t % self.test_freq_t == 0:
                self.compute_test_score()

    def run_MACES(self):
        """
        Run the multi-armed coreset selection algorithm.
        """
        t   = 0

        # collect T observations
        while t < self.T:

            # sample from reward dists
            pi              = self.sample_reward_probabilities()

            # transfer hidden -> train indices
            j               = self.sample_batch(pi, self.batch_size)

            # reward & belief update
            r               = self.compute_reward()
            self.update_beta_params(reward=r, j=j)

            # update counters
            t               += self.batch_size

            # test score
            if t % self.test_freq_t == 0: 
                self.compute_test_score()
                self.t_v_shuffle()

    def plot_scores(self, times:np.ndarray, scores: np.ndarray, *args, **kwargs):
        """
        After an algorithm is run, plot the scores over time.

        Args:
            times: list of times
            scores: list of scores
            *args: parameters for plt.plot()
            **kwargs: parameters for plt.plot()
        """
    
        plt.plot(times, scores, markersize=2, *args, **kwargs)
        plt.xlabel("Coreset size")
        plt.ylabel("Mean squared prediction error")
        plt.title(f"Test performance of model trained using coreset")

    def plot_beta_dist(self, feature: str, *args, **kwargs):
        """
        After run_MACES() is executed, plot the beta distributions for all categories of a given feature. 

        Args:
            *args: parameters for plt.plot()
            **kwargs: parameters for plt.plot()
        """
        # filter to relevant features & categories
        feature_name    = f"cluster_ID_{feature}"
        j_indices       = [index for index, (name, _) in enumerate(self.clusters) if name == feature_name]
        categories      = [cat for (name, cat) in self.clusters if name == feature_name]

        # get relevant alphas & betas
        alphas          = self.alphas[j_indices]
        betas           = self.betas[j_indices]

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

    def eval_test_performance(self, n_runs:int = 1, which: str = "MACES") -> np.ndarray:
        """
        Compute average test scores over n_runs for MACES or one of the baselines.

        Args:
            n_runs (int): number of times to repeat the algorithms
            which (str): whether to evaluate 'MACES', 'rb', 'TORRENT' or 'full'
        
        Returns:
            avg_scores (np.ndarray): score at every test time
        """
        current_run = 1
        avg_scores  = np.zeros(len(self.test_times))

        while current_run <= n_runs:
            
            print(f"Benchmarking run {current_run} for model {which}")

            torch.manual_seed(current_run + 100)
            np.random.seed(current_run + 100)
            random.seed(current_run + 100)

            self.reset()
            if which == 'MACES': self.run_MACES()
            elif which == 'rb': self.run_random_baseline()
            elif which == 'full':
                if isinstance(self.model, MLP):
                    # adapt lr for full model
                    self.model.lr       /= 100 #1e-2 -> 1e-4
                    self.model.init_lr  /= 100
                    self.run_full_model()
                    self.model.lr       *= 100
                    self.model.init_lr  *= 100
            elif which == 'TORRENT': self.run_TORRENT()
            elif which == 'KCG': self.run_k_centers_greedy()
            elif which == 'LL': self.run_LL()
            elif which == 'mds': self.run_metadist_selector()

            test_scores     = self.test_scores[:]
            train_indices   = self.train_indices[:]
            avg_scores      = np.sum((avg_scores, self.test_scores), axis=0)

            self.store_results(which, current_run, test_scores, train_indices)

            # if KCG, then store results early
            if which == 'KCG':
                with open(rf"C:\Users\nial\Documents\GitHub\Master-Thesis\Comprehensive Benchmarks\KCG_runs\run_{current_run}.pkl", "wb") as f:
                    pkl.dump((current_run, test_scores, train_indices), f)

            current_run += 1

        avg_scores      /= n_runs

        return avg_scores
    
    def plot_test_performance(self, avg_scores_dict: dict, ylim: tuple, n_runs: int):
        """
        Plot outputs from eval_test_performance().

        Args:
            avg_scores_dict: {label: avg_scores}
            ylim: (y_min, y_max)
            n_runs: number of runs of eval_test_performance()
        """
        plt.figure()
        colors = {'Full dataset':'black', 'Random selector':'cornflowerblue', 'TORRENT':'palegoldenrod', 'KCG': 'fuchsia', 'MACES':'indianred', 'Learning Loss':'lime', 'Metadata selector': 'darkorange'}
        
        assert all(key in colors.keys() for key in avg_scores_dict.keys()), '`plot_test_performance` received unknown method'

        for label, avg_scores in avg_scores_dict.items():
            linestyle   = '--' if label in ['Full dataset', 'TORRENT'] else '-'
            self.plot_scores(times=self.test_times, scores=avg_scores, label=label, color=colors[label], linestyle=linestyle, linewidth=2.5)

        # plt.ylim(ylim)
        plt.legend()
        plt.suptitle(f"N runs = {n_runs}")
        plt.show()
    
    def benchmark_MACES(self, n_runs:int = 1):
        """
        Plot average test scores over n_runs for random selector, full model and MACES.

        Args:
            n_runs: number of times to repeat the algorithms
        """
        # random baseline, MACES with all features and full model
        avg_scores_full             = self.eval_test_performance(n_runs, "full")
        avg_scores_rb               = self.eval_test_performance(n_runs, "rb")
        avg_scores_TORRENT          = self.eval_test_performance(n_runs, "TORRENT")
        # avg_scores_KCG              = self.eval_test_performance(n_runs, "KCG") TODO
        avg_scores_mds              = self.eval_test_performance(n_runs, 'mds')

        if isinstance(self.model, MLP):   
            avg_scores_LL           = self.eval_test_performance(n_runs, "LL")

        avg_scores_MACES             = self.eval_test_performance(n_runs, "MACES")

        avg_scores_dict             =  {'Full dataset': avg_scores_full, 'Random selector': avg_scores_rb\
                                   , 'TORRENT': avg_scores_TORRENT, 'MACES': avg_scores_MACES\
                                    , 'Metadata selector': avg_scores_mds} # 'KCG': avg_scores_KCG, 
        
        if isinstance(self.model, MLP): 
            avg_scores_dict['Learning Loss']\
                                    = avg_scores_LL 

        terminal_scores_dict        = {key:value[-1] for key, value in avg_scores_dict.items()}

        self.avg_scores_dict        = avg_scores_dict
        self.terminal_scores_dict   = terminal_scores_dict

        # each individual feature
        if len(self.features) > 1:

            for feature, n_bins in self.features.items():

                continue # TODO

                # generate cluster
                self.clean_clusters()
                self.generate_cluster(feature, n_bins)

                avg_scores_f    = self.eval_test_performance(n_runs, "MACES")

                avg_scores_dict[f"MACES {feature}"] \
                                =  avg_scores_f

        y_min                   = min([np.min(value) for _, value in avg_scores_dict.items()])
        y_max                   = max([np.max(value) for _, value in avg_scores_dict.items()])
        
        self.plot_test_performance(avg_scores_dict, (y_min, y_max), n_runs)
    
    def store_results(self, which: str, current_run:int, test_scores:list, train_indices: list):
        """
        Store results for one run of `eval_test_performance()`.

        Args:
            which (str): name of method run
            current_run (int): index of current run
            test_scores (list): scores of method at test times
            train_indices (list): indices of observations in coreset
        """
        # save on storage
        if which == 'full': 
            train_indices   = list()

        results_dict_i      = {'which': which, 'current_run': current_run, 'test_times':self.test_times,
                              'test_scores': test_scores, 'train_indices': train_indices}
        
        if which == 'MACES': 
            results_dict_i['sampled_C']         = self.sampled_C[:]

        self.results_dict[self.results_idx]     = results_dict_i
        
        self.results_idx                        += 1

    def __repr__(self):
        """
        String representation of instantiated object.

        Returns:
            repr (str): string of object call
        """
        class_name = self.__class__.__name__
        return f"""
        {class_name}(
        data, 
        x={self.x!r}, 
        y={self.y!r}, 
        features={self.features!r},
        T={self.T!r}, 
        batch_size={self.batch_size!r}, 
        test_freq={self.test_freq!r}, 
        test_times={self.test_times!r},
        model={self.model.__repr__()})
        """

# BANDIT ADVERSARIALLY ATTACKS TRAINING DATA

class crafty_bandit(bandit):
    """
    Corrupts training data and attempts to successfully model regardless.
    """
    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict, T: int=1000, batch_size: float=10\
                 , frac_train: float=0.5, frac_test: float=0.48, frac_val: float=0.02, test_freq: int=5, p_corrupt: float=0.1
                 , model=MLP()):
        """  
        Args:
            dataset: includes columns x and y
            x: name(s) of independent variable(s)
            y: column name of dependent variable
            features: along which to cluster df, including n_bins if numeric
            T: number of train points to sample before terminating (must be < frac_train * len(dataset))
            batch_size: number of points to sample before computing reward
            frac_train: fraction of dataset for training
            frac_test: fraction of dataset for testing
            frac_val: fraction of dataset for validation
            test_freq: frequency (in batches) at which to evaluate the model fit on test set
            p_corrupt: proportion of training observations to corrupt
        """

        super().__init__(dataset=dataset, x=x, y=y, features=features, T=T, batch_size=batch_size\
                 , frac_train=frac_train, frac_test=frac_test, frac_val=frac_val, test_freq=test_freq, model=model)

        # clusters and TTV split
        self.ttv_split()
        self.corrupt_train_data(p_corrupt)
        self.clean_clusters()
        self.generate_clusters(self.features)

    def corrupt_train_data(self, p_corrupt: float):
        """
        Corrupt training datapoints Y and record.

        Args:
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

# ND BANDIT PERMITS N-DIMENSIONAL CLASSES

class ND_bandit(bandit):
    """
    Bandit permitting n-dimensional classes (i.e., comprising n features).
    All features are crossed to generate clusters.
    """

    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict\
                 , T: int=1000, batch_size: float=10, frac_train: float=0.5, frac_test: float=0.48\
                 , frac_val: float=0.02, test_freq: int=5, model = MLP()):
        """
        Args:
            dataset: includes columns x and y
            x: name(s) of independent variable(s)
            y: column name of dependent variable
            features: along which to cluster df, including n_bins if numeric
            T: number of train points to sample before terminating (must be < frac_train * len(dataset))
            batch_size: number of points to sample before computing reward
            frac_train: fraction of dataset for training
            frac_test: fraction of dataset for testing
            frac_val: fraction of dataset for validation
            test_freq: frequency (in batches) at which to evaluate the model fit on test set
            model: object with fit() and predict() methods
        """
  
        # instantiate for mapping clusters to features
        self.cluster_dict       = dict()

        super().__init__(dataset=dataset, x=x, y=y, features=features, T=T, batch_size=batch_size, frac_train=frac_train\
                         , frac_test=frac_test, frac_val=frac_val, test_freq=test_freq, model=model)

    def generate_clusters(self, features: dict):
        """
        Generate a cluster for every entry in the cartesian product of binned features.

        Args:
            features: {feature: n_bins} dictionary

        Returns:
            Column 'ND_cluster' added to self.dataset indicating cluster membership.
        """
        # generate unit clusters
        super().generate_clusters(features)

        # unique combinations
        dataset_clusters                        = self.dataset.filter(like='cluster_ID_')
        unique_combinations                     = dataset_clusters.drop_duplicates().copy(deep=True) 
        unique_combinations.reset_index(drop=True, inplace=True)

        # map cluster to feature values
        for idx, row in unique_combinations.iterrows():

            row_dict                            = row.to_dict()
            self.cluster_dict[idx]              = row_dict
        

        # 1D -> ND clusters
        unique_combinations['ND_cluster']       = unique_combinations.index
        dataset_merged                          = pd.merge(self.dataset, unique_combinations, on=list(dataset_clusters.columns), how='left')
        dataset_merged.drop(dataset_clusters.columns, axis=1, inplace=True)
        dataset_merged.set_index(self.dataset.index, inplace=True) # maintain original index

        # delete unit clusters
        self.clean_clusters()

        # retain ND clusters
        self.clusters                           = [('ND_cluster', value) for value in unique_combinations['ND_cluster'].unique()]
        self.dataset                            = dataset_merged
        self.cluster_sizes                      = self.dataset[self.dataset.index.isin(self.hidden_indices)]['ND_cluster'].value_counts()

    def under_the_hood(self, pi:np.ndarray, j:int, current_score: float, prev_score: float, r:float):
        """
        Provide intermittent status reports about the agent during data selection.
        Only valuable when batch_size == 1. Not enabled by default.
        plot_beta_dist() call suppressed: too much customizability.
         
        Args:
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
        After run_MACES() is executed, plot the beta distributions at a fixed value of a feature.

        Args:
            feature: to fix
            fixed_val: fixed value of feature
            *args: parameters for plt.plot()
            **kwargs: parameters for plt.plot()
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

    def store_results(self, which: str, current_run:int, test_scores:list, train_indices: list):
        """
        Store results for one run of `eval_test_performance()`.

        Args:
            which (str): name of method run
            current_run (int): index of current run
            test_scores (list): scores of method at test times
            train_indices (list): indices of observations in coreset
        """
        # save on storage
        if which == 'full': 
            train_indices   = list()

        results_dict_i      = {'which': which, 'current_run': current_run, 'test_times':self.test_times,
                              'test_scores': test_scores, 'train_indices': train_indices}
        
        if which == 'MACES':
            results_dict_i['sampled_C'] = self.sampled_C[:]
            results_dict_i['cluster_dict'] = self.cluster_dict
            results_dict_i['cluster_sizes'] = self.cluster_sizes

        self.results_dict[self.results_idx]     = results_dict_i
        
        self.results_idx                        += 1

# LAZY BANDIT DOES NOT PERFORM TTV SPLIT

class lazy_bandit(ND_bandit):
    """
    Like bandit, but too lazy to shuffle the data. 
    User specifies hidden (train), validation and test indices.
    """
    
    def __init__(self, dataset: pd.DataFrame, x: str, y: str, features: dict\
                 , hidden_indices: list, test_indices: list, val_indices: list\
                 , T: int=1000, batch_size: float=10, test_freq: int=5, model= MLP()):
        
        """
        Args:
            dataset: includes columns x and y
            x: name(s) of independent variable(s)
            y: column name of dependent variable
            features: along which to cluster df, including n_bins if numeric
            T: number of train points to sample before terminating (must be < frac_train * len(dataset))
            batch_size: number of points to sample before computing reward
            hidden_indices: indices of datapoints eligible for training
            test_indices: indices of datapoints for testing
            val_indices: indices of datapoints for validation
            test_freq: frequency (in batches) at which to evaluate the model fit on test set
            model: object with fit() and predict() methods
        """
        # copy to avoid mutation issues
        self.init_hidden_indices= hidden_indices[:]
        self.hidden_indices     = hidden_indices
        self.test_indices       = test_indices
        self.val_indices        = val_indices
        self.train_indices      = []

        super().__init__(dataset, x, y, features, T, batch_size, 0, 0, 0, test_freq, model)

    def ttv_split(self):
        """
        Overwrite ttv_split() to do nothing
        """
        pass

    def t_v_shuffle(self):
        """
        Overwrite t_v_shuffle() to do nothing
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

    Args:
        shuffle: True or False

    Returns:
        instance of class with shuffle implemented (or not)
    """

    def class_shuffler(cls):
        """
        Args:
            cls: class

        Returns:
            class with desired parent
        """

        parent = bandit if shuffle else lazy_bandit

        class shuffle_child(parent):
            pass
            
        return shuffle_child
    
    return class_shuffler
