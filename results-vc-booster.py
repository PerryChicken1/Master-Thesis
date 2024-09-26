import numpy as np
import pandas as pd
import random
import pickle as pkl
import os
import pandas.api.types as ptypes
from sklearn.linear_model import LinearRegression
from MACES import lazy_bandit
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from vc_boost_method import VCBooster
from vc_boost_loss import LS
from custom_ttv_split import load_data, get_ttv_indices

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


def benchmark_vc_booster(df_global: pd.DataFrame, VCBooster: VCBooster, x: list | str, y: list | str, features: dict, T: int=4000, n_runs:int=10, run_MACES: bool=True) -> dict:
    """
    Benchmark varying coefficient model constructed from gradient boosting vs MACES.

    VC formulation: Y = X * B(Z). Model coefficients depend on metadata Z.
    MACES formulation: Y = X(Z) * B. Choice of training observations depends on metadata Z.
    
    Args:
        df_global (pd.DataFrame): output from `load_data()`
        VCBooster (VCBooster): instance of VCBooster class (credit: sply88 on Github)
        x (list | str): column name(s) of predictor(s) in `df_global`
        y (list | str): column name(s) of target(s) in `df_global`
        features (dict): metadata features to pass to MACES
        T (int): n. observations to include in model `fit()` call
        n_runs: number of independent runs over which to average test scores
        run_MACES (bool): whether to additionally run MACES

    Returns:
        bmk_results (dict): test scores for the methods for each run 
    """
    # dict to store results
    bmk_results = dict()

    # add dummy variables to `df_global`, if there are categorical variables
    df_global   = dummy_encode(df_global, features)
    
    # effect modifiers
    z           = [key for key, value in features.items() if value is not None]
    z.extend([c for c in df_global.columns if 'dummy_' in c])

    # split df
    df_train    = df_global[df_global['new_colour'] == "train"]
    df_test     = df_global[df_global['new_colour'] == "test"]
    df_val      = df_global[df_global['new_colour'] == "val"]

    # validation data for VCBooster
    X_val       = df_val[x].to_numpy()
    Y_val       = df_val[y].to_numpy()
    Z_val       = df_val[z].to_numpy()

    # test data for VCBooster
    X_test      = df_test[x].to_numpy()
    Y_test      = df_test[y].to_numpy()
    Z_test      = df_test[z].to_numpy()

    # MACES arguments
    hidden_indices, test_indices, val_indices\
                =  get_ttv_indices(df_global, 'new_colour')

    for i in range(n_runs):

        # HOLDOUT TEST SCORE USING VARYING COEFFICIENT MODEL WITH METADATA AS PREDICTORS

        print(f"Evaluation {i+1} / {n_runs} of VCBooster")

        # fix seeds for reproducibility
        random.seed(i)
        np.random.seed(i)

        # obtain data subset for model fitting
        df_subset   = df_train.sample(T)
        X_subset    = df_subset[x].to_numpy()
        Y_subset    = df_subset[y].to_numpy()
        Z_subset    = df_subset[z].to_numpy()

        # fit & evaluate VCBooster
        model_vc_i  = VCBooster
        model_vc_i  = model_vc_i.fit(X_subset, Z_subset, Y_subset, (X_val, Z_val, Y_val))
        y_pred_i    = model_vc_i.predict(X_test, Z_test)
        mse_i       = mean_squared_error(Y_test, y_pred_i)

        bmk_results[('VC', i)]      = mse_i
        bmk_results[('VC_booster', i)]\
                                    = model_vc_i

        # HOLDOUT TEST SCORE BY APPLYING LINEAR REGRESSION WITH METADATA ONLY USED AS SELECTION CRITERION
        if run_MACES:
        
            print(f"Evaluation {i+1} / {n_runs} of MACES. T = {T}")

            MACES       = lazy_bandit(df_global, x, y, features, hidden_indices, test_indices, val_indices, T, 1, 100, LinearRegression())
            MACES.run_MACES()

            bmk_results[('MACES_terminal_test_score', i)]   = MACES.test_scores[-1]
            bmk_results[('MACES_sampled_C', i)]             = MACES.sampled_C
            bmk_results[('MACES_alphas', i)]                = MACES.alphas
            bmk_results[('MACES_betas', i)]                 = MACES.betas
            bmk_results[('MACES_rewards', i)]               = MACES.rewards
    
    return bmk_results

if __name__ == '__main__':

    #######################################################################################
    # SPECIFY PARAMETERS                                                                  #
    #######################################################################################

    # number of datapoints to include in training data
    T           = 4000

    # independent runs
    n_runs      = 10

    # load dataframe
    df_global   = load_data()

    # specify predictors (x), target (y) and effect modifiers with # quantiles (features)
    x           = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
    y           = "agbd"
    features    = {"elev_lowes": 3, "pft_class": None, "region_cla": None}

    # effect modifiers are the metadata features theoretically available at global scale

    #######################################################################################
    # COMPARE PERFORMANCE                                                                 # 
    #######################################################################################

    # evaluate methods
    bmk_results = benchmark_vc_booster(df_global, VCBooster(verbose=0), x, y, features, T, n_runs, True)

    # ensure filepath exists
    fpath       = rf"C:\Users\nial\Documents\GitHub\Master-Thesis\Comprehensive Benchmarks\vcboost_\vcboost_.pkl"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    # store test scores
    with open(fpath, 'wb') as f:
        pkl.dump(bmk_results, f)
