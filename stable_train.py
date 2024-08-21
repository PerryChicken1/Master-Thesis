# function(s) to assess the stability of a neural network
# background: there is currently high variance in the full model score between runs

import pandas as pd
import random
import torch
from torch import nn
from custom_models import MLP


def assess_stability(dataset:pd.DataFrame, x: str | list, y: str='agbd', colour_col: str='new_colour', n_seed: int=10, **kwargs):
    """
    Train MLP model on dataset with different random seeds. 

    INPUTS:
    dataset: dataframe with predictors, target values and TTV split
    x: predictor(s)
    y: target
    colour_col: TTV split indicator
    n_seed: number of random seeds to test
    kwargs: arguments passed to instantiate MLP
    """
    test_scores     = dict()

    # split data
    X_train         = dataset[dataset[colour_col] == 'train'][x].to_numpy()
    y_train         = dataset[dataset[colour_col] == 'train'][y].to_numpy()
    X_test          = dataset[dataset[colour_col] == 'test'][x].to_numpy()
    y_test          = dataset[dataset[colour_col] == 'test'][y].to_numpy()

    # tensorize
    X_test          = torch.tensor(X_test, dtype=torch.float32)
    y_test          = torch.tensor(y_test, dtype=torch.float32)

    # fit model with different seeds
    for i in range(n_seed):

        print(f"Fitting model {i+1}")

        random.seed(i)
        
        model           = MLP(**kwargs)

        model.fit(X_train, y_train)

        y_pred          = model.predict(X_test)
        test_score      = model.loss_fn(y_test, y_pred)
        test_scores[i]  = test_score

        print(f"Test score {i}: {test_score}")

    return test_scores


