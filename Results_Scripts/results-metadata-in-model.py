import numpy as np
import pandas as pd
import pickle as pkl
import pandas.api.types as ptypes
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import MinMaxScaler
from MACES_and_Benchmark_Methods.MACES import lazy_bandit
from Models_and_Helper_Functions.benchmarking_utils import tabulate_bmk_outputs, plot_test_times
from Models_and_Helper_Functions.custom_ttv_split import load_data, get_ttv_indices
from Models_and_Helper_Functions.custom_models import MLP
from torch import nn

# Run as __main__ to reproduce results for section 4.2, Poisson regression task
# Compares metadata in model vs metadata for coreset selection


def dummy_encode(df_global: pd.DataFrame, features: dict):
    """
    Encode categorical variables in `df_global` as dummies.

    Args:
        df_global (pd.DataFrame): dataset containing effect modifiers in `features.keys()` as columns.
        features (dict): dictionary of {feature_column: number categories}
    
    Returns:
        df_global (pd.DataFrame): dataset with dummy variable columns added.
    """
    assert not any([col.__contains__('dummy_') for col in df_global.columns]), 'Dummy encoding already provided'

    for feature in features.keys():

        # if categorical
        if ptypes.is_categorical_dtype(df_global[feature]):
            
            dummies     = pd.get_dummies(df_global[feature], prefix='dummy_' + feature, drop_first=True)
            df_global   = pd.concat([df_global, dummies], axis=1)
    
    return df_global


# evaluate the MLP when metadata are included as predictors alongside x

def model_with_metadata_bmk(df_global: pd.DataFrame, x: list | str, y: str, features: dict, hidden_indices: list, test_indices: list, val_indices: list 
                               , T: int=4000, batch_size: int=10, test_freq: int=10, n_runs: int=10,
                               filename: str='results-metadata-in-model', description: str='Evaluating MLP(m, x)'):
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
    os.makedirs(folder, exist_ok=True)
    plot_title      = description

    # dummy encode categorical variables
    df_global       = dummy_encode(df_global, features)

    # scale elevation to (0,1)
    # quick and dirty approach :')
    scaler                      = MinMaxScaler()
    df_global['elev_lowes']     = scaler.fit_transform(df_global[['elev_lowes']])


    # add metadata to predictors
    if isinstance(x, str): 
        x           = [x]
    
    x_original      = x[:]

    #   add dummy columns
    x               = x + [m for m in df_global.columns if 'dummy_' in m]

    #   add numeric metadata
    x               = x + [m for m in features.keys() if ptypes.is_numeric_dtype(df_global[m])]
    
    #   print predictors
    print("x = ", x)

    # instantiate bandit
    model           = PoissonRegressor()
    bandit          = lazy_bandit(df_global, x, y, features, hidden_indices, test_indices, val_indices, T, batch_size, test_freq, model) 

    # evaluate test performance with random selection + metadata
    bandit.eval_test_performance(n_runs, 'rb')

    # evaluate test performace with MACES + metadata
    bandit.eval_test_performance(n_runs, 'MACES')
    results_dict    = bandit.results_dict

    # evaluate test performance with MACES, no metadata
    print("x_original =", x_original)
    bandit          = lazy_bandit(df_global, x_original, y, features, hidden_indices, test_indices, val_indices, T, batch_size, test_freq, model) 
    bandit.eval_test_performance(n_runs, 'MACES')
    results_dict_2  = bandit.results_dict

    # merge results dictionaries into one
    key_offset      = max(results_dict.keys()) + 1
    results_dict_2  = {(key + key_offset ):value for key, value in results_dict_2.items()}
    
    for key, value in results_dict.items():
        if value['which']  == 'MACES':
            results_dict[key]['which']  = 'MACES-with-m'

    results_dict    = results_dict | results_dict_2

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
    features                = {"elev_lowes": 3, "pft_class": None, "region_cla": None}

    # MACES parameters
    T                       = 4000
    batch_size              = 10
    test_freq               = 10
    n_runs                  = 10

    # model
    model                   = PoissonRegressor()

    # agent
    bandit_global           = lazy_bandit(dataset=df_global, x=x, y=y, features=features, hidden_indices=hidden_indices
                                      , test_indices=test_indices, val_indices=val_indices, T=T, batch_size=batch_size
                                      , test_freq=test_freq, model=model)
    
    filename                = 'results-trio-poisson'
    description             = 'Evaluating Poisson(m, x)'

    ####################################################################
    # RUN                                                              #
    ####################################################################

    model_with_metadata_bmk(df_global, x, y, features, hidden_indices, test_indices, val_indices, T, batch_size, test_freq
                            , n_runs, filename=filename, description=description)