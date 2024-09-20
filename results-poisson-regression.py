import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from MACES import lazy_bandit
from benchmarking_utils import comprehensive_benchmark
from custom_ttv_split import load_data, get_ttv_indices
from torch import nn

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

# MABS parameters
T                       = 4000
batch_size              = 10
test_freq               = 10

# model
model                   = PoissonRegressor()

# agent
bandit_global           = lazy_bandit(dataset=df_global, x=x, y=y, features=features, hidden_indices=hidden_indices
                                      , test_indices=test_indices, val_indices=val_indices, T=T, batch_size=batch_size
                                      , test_freq=test_freq, model=model)

####################################################################
# RUN                                                              #
####################################################################

comprehensive_benchmark(lazy_bandit_=bandit_global, 
                        description='Poisson Regressor Comprehensive Bmk',
                        filename='poisson-regressor-bmk',
                        n_runs=10
                        )