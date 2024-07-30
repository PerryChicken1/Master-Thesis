import contextlib
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
import numpy as np
import os
import itertools
from MABS import lazy_bandit
from random import shuffle

def ensure_dir(file_path):
    """
    Create directory from filepath if it does not exist.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def benchmark_bandit(bandit, n_runs: int, description: str=None):
    """
    Benchmark MABS on bandit and store results.

    INPUTS:
    bandit: instance of bandit class or a bandit subclass
    n_runs: average test scores over n runs
    description: optionally, a description of the particular benchmark run
    """
    # file name based on date & time
    fpath           = r"C:\Users\nial\Documents\GitHub\Master-Thesis\Results"
    dtime           = datetime.datetime.now()
    dtime           = dtime.strftime("%d-%m-%Y, %H.%M.%S")
    folder          = rf"{fpath}\{dtime}"
    filename        = rf"{folder}\Benchmark_plot.png"

    # save comparison plot
    @contextlib.contextmanager
    def save_plot_as_png(filename):
    
        # function to replace
        original_show = plt.show
    
        # new function
        def save_figure(*args, **kwargs):
            plt.savefig(filename)
    
        # replacement
        plt.show = save_figure

        # directory
        ensure_dir(filename)
    
        # execute code with replacement
        yield
    
        # undo replacement
        plt.show = original_show

    with save_plot_as_png(filename):
        bandit.benchmark_MABS(n_runs)
    
    # prompt user for description
    if not description: 
        description = input("Briefly describe this benchmark run:")

    # store specifications
    specs           = {"Description": description
                       , "Call": bandit.__repr__()
                       , "Terminal test scores": bandit.terminal_scores_dict
                       , "Average test scores": bandit.avg_scores_dict
                       , "N runs": n_runs
                       }

    with open(folder + r"\Specs.pkl", "wb") as specs_file:
        pkl.dump(specs, specs_file)

def benchmark_subset_n(bandit, n: int=4000, n_runs: int=1):
    """
    Reduce the hidden set to n observations, then benchmark the methods.
    Key: terminal score of full model is of interest here. Do we get ~9000 as with rbs or do we get < 8000 as with full model?

    bandit.dataset should contain more than n hidden points

    INPUTS:
    bandit: instance of bandit() class, or one of the subclasses
    n: number of hidden indices
    n_runs: argument to benchmark_MABS()
    """

    hidden_indices              = bandit.hidden_indices
    hidden_indices_n            = np.random.choice(hidden_indices, size=n, replace=False)

    bandit.init_hidden_indices  = hidden_indices_n
    bandit.hidden_indices       = hidden_indices_n

    bandit.benchmark_MABS(n_runs=n_runs)

def load_specs(DD: str, MM: str, YYYY:str,  h: str, m: str, s: str):
    """
    Load specification dictionary for benchmarking run at specified date & time.

    INPUTS:
    DD: day of month
    MM: month (01 - 12)
    YYYY: year (2024)
    h: hour (00-23)
    m: minute (00-59)
    s: second (00-59)

    OUTPUTS:
    specs: dictionary of run specifications
    """
    with open(rf"C:\Users\nial\Documents\GitHub\Master-Thesis\Results\{DD}-{MM}-{YYYY}, {h}.{m}.{s}\Specs.pkl", 'rb') as f:
        specs = pkl.load(f)

    return specs

def lazy_bandit_feature_search(features: dict, model, tuple_size: int=3, n_runs: int=10, **kwargs):
    """
    Benchmark MABS method for several feature combinations using ND bandit.

    INPUTS:
    features: superset of features to choose from
    model: model with fit() and predict() methods to assign to bandit; inc. hyperparameters specified 
    tuple_size: number of features to pass to the agent. <= len(features)
    n_runs: argument to benchmark_MABS()
    kwargs: other arguments to bandit()
    """
    # all combinations of size `tuple_size`
    feat_names          = list(features.keys())
    feat_combinations   = list(itertools.combinations(feat_names, tuple_size))
    shuffle(feat_combinations)

    # benchmark MABS for each combination
    for combination in feat_combinations:
        
        # select features
        features_c      = dict((feat_name, features[feat_name]) for feat_name in combination)

        # instantiate and benchmark bandit
        bandit          = lazy_bandit(features=features_c, **kwargs)
        bandit.model    = model
        description     = ", ".join(features_c.keys())
        benchmark_bandit(bandit=bandit, n_runs=n_runs, description=description)