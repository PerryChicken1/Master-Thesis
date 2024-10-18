import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from Models_and_Helper_Functions.custom_models import MLP
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn import functional as F
from MACES_and_Benchmark_Methods.MACES import lazy_bandit

# Function(s) to assess the stability of a neural network
# Was used to investigate & prevent overfitting


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

def compare_lr(bandit_global: lazy_bandit, lr_list: list=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], n_runs: int=5) -> dict:
    """
    Compare the test performance of the full model at different learning rates.

    Args: 
        bandit_global (lazy_bandit): `lazy_bandit` with `MLP` model attribute
        lr_list (list): learning rates to compare

    Returns:
        test_scores (dict): {lr: test scores}
    """
    test_scores = dict()

    for lr in lr_list:
        bandit_global.reset()
        bandit_global.model.init_lr = lr
        bandit_global.model.lr      = lr
        avg_scores                  = bandit_global.eval_test_performance(n_runs, which='full')
        test_scores[lr]             = avg_scores

    return test_scores


class MLP_test(MLP):
    """
    `MLP` that tracks the test loss during `fit()` call.
    Used to select hyperparameters that best suit the full model (i.e., prevent overfitting).
    """

    def fit(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Run training loop at time t.

        Args:
            X: data
            y: labels
            X_test: test data
            y_test test labels
        """
        # clean slate
        self.reset_model_parameters()

        # DataLoader
        X_tensor        = self.tensorize(X)
        y_tensor        = self.tensorize(y)
        X_test_tensor   = self.tensorize(X_test)
        y_test_tensor   = self.tensorize(y_test)

        tensor_dataset  = TensorDataset(X_tensor, y_tensor)
        train_loader    = DataLoader(dataset=tensor_dataset, batch_size=32, shuffle=True)

        self.train()

        optimiser       = Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        loss_fn         = self.loss_fn

        if self.with_scheduler: 
            factor      = 0.5
            patience    = 3
            min_lr      = self.init_lr * (factor ** 4) 
            scheduler   = lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=factor, patience=patience, min_lr=min_lr)
        
        self.fit_count += 1

        # train
        for epoch in range(self.num_epochs):
        
            epoch_loss  = 0
            n_iter      = 0

            for batch in train_loader:

                X_batch, y_batch    = batch

                optimiser.zero_grad()

                outputs             = self.forward(X_batch)
                
                loss_batch          = loss_fn(y_batch, outputs)

                loss_batch.backward()
                optimiser.step()

                epoch_loss          += loss_batch.item()
                n_iter              += 1

            epoch_loss /= n_iter

            if self.with_scheduler: 
                scheduler.step(epoch_loss)
                self.lr     = scheduler.get_last_lr()[0] # track lr

            y_test_hat      = self.forward(X_test_tensor)
            test_loss       = loss_fn(y_test_hat, y_test_tensor)    

            # intermittent updates
            if self.fit_count % self.print_freq == 1: 
                print(f'Model fit {self.fit_count}. Epoch [{epoch+1}/{self.num_epochs}], Training loss: {epoch_loss:.4f}, Test loss: {test_loss: .4f}, lr: {self.lr}')
