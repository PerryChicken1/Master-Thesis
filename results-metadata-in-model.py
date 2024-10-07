import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import mean_squared_error
from MACES import lazy_bandit
from benchmarking_utils import tabulate_bmk_outputs, plot_test_times
from custom_ttv_split import load_data, get_ttv_indices
from custom_models import MLP
from torch import nn

# evaluate the MLP when metadata are included as predictors alongside x

def model_with_metadata_bmk(df_global: pd.DataFrame, x: list | str, y: str, features: dict, hidden_indices: list, test_indices: list, val_indices: list 
                               , T: int=4000, batch_size: int=10, test_freq: int=10, n_runs: int=10
                               , num_epochs: int=5, lr: float=0.01, print_freq: int=10, with_scheduler: bool=False, loss_fn: nn.Module=nn.MSELoss(), filename: str='results-metadata-in-model'
                               , description: str='Evaluating MLP(m, x)'):
    """
    Function to compare inputting metadata into the MLP vs coreset selection using the metadata.

    Args:
        df_global (pd.DataFrame): data for modelling
        x (list | str): model predictor(s)
        y (str): name of target column in `df_global`
        features (dict): dictionary of {feature_name: number of bins}
        T (int): coreset size
        batch_size (int): batch size for MACES
        test_freq (int): test frequency in batches
        n_runs (int): number of independent evaluation runs
        num_epochs (int): number of epochs for training
        lr (float): learning rate for `MLP`
        print_freq (int): frequency of printing updates (in # test batches)
        with_scheduler (bool): whether to use lr scheduler in training
        loss_fn (nn.module): neural network loss function
        filename (str): desired filename for outputs
    """
    # folder to store results
    folder          = rf"C:\Users\nial\Documents\GitHub\Master-Thesis\Comprehensive Benchmarks\{filename}"
    plot_title      = 'MLP(m, x) with random selector'
    
    # add metadata to predictors
    if isinstance(x, str): 
        x           =   [x]

    x               = x + [m for m in features.keys()]
    n_predictors    = len(x)

    # instantiate bandit
    model           = MLP(n_predictors, num_epochs, lr, print_freq, with_scheduler, loss_fn)
    bandit          = lazy_bandit(df_global, x, y, features, hidden_indices, test_indices, val_indices, T, batch_size, test_freq, model) 

    # evaluate test performance with random selection + metadata
    bandit.eval_test_performance(n_runs, 'rb')
    results_dict    = bandit.results_dict

    # store results
    lazy_bandit_repr= bandit.__repr__()

    with open(folder + rf"\{filename}.pkl", "wb") as specs_file:
        pkl.dump((description, lazy_bandit_repr, results_dict), specs_file)
           
    bmk_table      = tabulate_bmk_outputs(filename, average=False, dump=True)

    plot_test_times(bmk_table, title=plot_title, filename=filename)

if __name__ == '__main__':

    ####################################################################
    # PARAMETERS                                                       #
    ####################################################################

    df_global               = load_data()

    hidden_indices, test_indices, val_indices\
                            =  get_ttv_indices(df_global, 'new_colour')

    # features
    x                       = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
    y                       = 'agbd'
    features                = {'region_cla':None, 'elev_lowes':3, 'selected_a': None}

    # MACES parameters
    T                       = 4000
    batch_size              = 10
    test_freq               = 10
    n_runs                  = 10

    # model
    n_predictors            = len(x)
    num_epochs              = 5
    lr                      = 0.01
    print_freq              = 10
    with_scheduler          = False
    loss_fn                 = nn.MSELoss()
    model                   = MLP(n_predictors, num_epochs, lr, print_freq, with_scheduler, loss_fn)

    # agent
    bandit_global           = lazy_bandit(dataset=df_global, x=x, y=y, features=features, hidden_indices=hidden_indices
                                      , test_indices=test_indices, val_indices=val_indices, T=T, batch_size=batch_size
                                      , test_freq=test_freq, model=model)
    
    filename                = 'results-metadata-in-model'
    description             = 'Evaluating MLP(m, x)'

    ####################################################################
    # RUN                                                              #
    ####################################################################

    model_with_metadata_bmk(df_global, x, y, features, hidden_indices, test_indices, val_indices, T, batch_size, test_freq
                            , n_runs, num_epochs, lr, print_freq, with_scheduler, loss_fn, filename)