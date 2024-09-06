import numpy as np
import torch
import random
from warnings import warn
from typing import Set, Optional, Self
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn import functional as F
from custom_score_functions import LogRatioLoss, NUGGET

# CUSTOM MODELS NEED fit(), predict() AND reinit_model() METHODS

# SUPERCLASS: RESETTABLE MODEL

class resettable_model(nn.Module):

    def __init__(self, state_dict_path: str):
        """
        Model can be reset to pre-train parameter values, reinitialised, or have architecture updated.

        Args:
            state_dict_path (str): absolute path where state_dict() is saved
        """
        super().__init__()
        self.path          = state_dict_path

    def forward(self, x):
        raise NotImplementedError("`forward()` method should be implemented by subclasses.")

    def save_initial_state(self):
        """
        Save initial parameters for later resets. 
        """
        torch.save(self.state_dict(), self.path)
    
    def update_architecture(self, new_architecture: nn.Sequential):
        """
        Update MLP architecture of self.

        Args:
            new_architecture: Sequential object of layers and activations
        """
        self.layers = new_architecture
        self.save_initial_state()
    
    def reset_model_parameters(self):
        """
        Reset the model parameters to original state dict.
        """
        self.load_state_dict(torch.load(self.path))
    
    @staticmethod
    def reinit_weights(m: nn.Linear):
        """
        Method to reinitialise weights of one layer in NN.
        """
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def reinit_model(self):
        """
        Reinitialise model parameters.
        """
        self.apply(self.reinit_weights)
        print(f"First 10 weights of layer 1: {self.state_dict()['layers.0.weight'][0][0:10]}")


# MULTI-LAYER PERCEPTRON 

class MLP(resettable_model):
    
    def __init__(self, n_predictors: int=12, num_epochs: int=5, lr: float=0.01\
                 , print_freq: int=10, with_scheduler=True, loss_fn=nn.MSELoss):

        """
        A simple three-layer perceptron model.

        Args:
            n_predictors: number of predictors
            num_epochs: for training at time t
            lr: initial learning rate
            print_freq: frequency (in MABS batches) at which to print updates
            with_scheduler: use learning rate scheduler?
            loss_fn: loss function to train model
        """

        super().__init__(r"C:\Users\nial\Documents\GitHub\Master-Thesis\State dict\MLP_state_dict.pt")

        self.n_predictors   = n_predictors 
        self.num_epochs     = num_epochs
        self.init_lr        = lr
        self.lr             = lr
        self.print_freq     = print_freq
        self.with_scheduler = with_scheduler
        self.loss_fn        = loss_fn
        self.fit_count      = 0

        self.layers         = self.build_model()
        self.save_initial_state()
        
    @staticmethod
    def tensorize(array:np.ndarray):
        """
        Convert numpy nd array into a tensor.

        Args:
            array: nd array

        Returns:
            tensor: tensor with float32s
        """
        if isinstance(array, torch.Tensor): return array
        else: return torch.tensor(array, dtype=torch.float32)

    def build_model(self):
        """
        Instantiate model layers.
        """
        layers = nn.Sequential(
                        nn.Linear(self.n_predictors, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Softplus()
                        )
        return layers
    
    def forward(self, x):
        output  = self.layers(x)
        return torch.flatten(output)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run training loop at time t.

        Args:
            X: data
            y: labels
        """
        # clean slate
        self.reset_model_parameters()

        # DataLoader
        X_tensor        = self.tensorize(X)
        y_tensor        = self.tensorize(y)

        tensor_dataset  = TensorDataset(X_tensor, y_tensor)
        train_loader    = DataLoader(dataset=tensor_dataset, batch_size=32, shuffle=True)

        self.train()

        optimiser       = Adam(self.parameters(), lr=self.lr)
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

            # intermittent updates
            if self.fit_count % self.print_freq == 1: 
                print(f'Model fit {self.fit_count}. Epoch [{epoch+1}/{self.num_epochs}], Training loss: {epoch_loss:.4f}, lr: {self.lr}')

    def predict(self, X: np.ndarray):
        """
        Predict with current model.
        """
        self.eval()

        X_tensor    = self.tensorize(X)

        with torch.no_grad():
            return self.forward(X_tensor)
        
    def reinit_model(self):
        """
        Reinitialise model parameters and attributes.
        """

        super().reinit_model()

        self.fit_count  = 0
        self.lr         = self.init_lr
        print(f"MLP model reinitialised")

    def __repr__(self):
        """
        Add input parameters to representation string.
        """
        super_repr      = super().__repr__()
        param_repr      = f"num_epochs = {self.num_epochs}, \n lr = {self.init_lr}, \n with_scheduler = {self.with_scheduler}"
        return super_repr + "\n" + param_repr

# MLP COMPATIBLE WITH LOSS PREDICTION

class MLP_loss_learn(MLP):

    def __init__(self, n_predictors: int=12, num_epochs: int=5, lr: float=0.01\
                 , print_freq: int=10, with_scheduler=True, loss_fn=nn.MSELoss):

        """
        Perceptron model with custom forward() method to acommodate jointly learning the loss.

        Source: https://arxiv.org/pdf/1905.03677

        Args:
            n_predictors: number of predictors
            num_epochs: for training at time t
            lr: initial learning rate
            print_freq: frequency (in MABS batches) at which to print updates
            with_scheduler: use learning rate scheduler?
            loss_fn: loss function to train model
        """ 
        super().__init__(n_predictors, num_epochs, lr, print_freq, with_scheduler, loss_fn)

    def forward(self, x):

        output          = x
        layer_outputs   = []

        for layer in self.layers:
            output = layer(output)

            # store results from intermediate activation layers
            if isinstance(layer, nn.ReLU):
                layer_outputs.append(output)

        return torch.flatten(output), layer_outputs

    def update_architecture(self, new_architecture: nn.Sequential):
        warn("`forward()` method assumes that intermediate activation functions are all `ReLU()`")
        super().update_architecture(new_architecture)
    
# LOSSNET PREDICTS THE LOSSES OF MLP

class LossNet(resettable_model):
    
    def __init__(self, embedding_sizes: list = [64, 32], FC_dim: int = 32):
        """
        Network for predicting MLP losses based on MLP embeddings.

        Args:
            n_layers: number of parameterised layers in MLP
            FC_dim: number of nodes in fully-connected layers 
        """

        super().__init__(r"C:\Users\nial\Documents\GitHub\Master-Thesis\State dict\LossNet_state_dict.pt")
        self.__dict__.update(vars())

        # FC layers (embeddings -> concatenation)
        self.FC_layers  = self._init_FC_layers(embedding_sizes, FC_dim)

        # final layer (concatenation -> loss prediction)
        self.linear     = nn.Linear(self.n_layers_ * FC_dim, 1)

        self.save_initial_state()

    @property
    def n_layers_(self):
        return len(self.embedding_sizes)
    
    @staticmethod
    def _init_FC_layers(embedding_sizes: list, FC_dim: int):
        """
        Instantiate sequence of fully-connected layers.
        
        Args:
            embedding_sizes: input sizes for layers
            FC_dim: output size for layers
        """
        FC_layers = nn.ModuleList()
        
        for e_size in embedding_sizes:
            FC_layers.append(nn.Linear(e_size, FC_dim))
        
        return FC_layers
    
    def forward(self, layer_outputs: list):
        """
        Args:
            layer_outputs: embeddings from MLP layers   
        """
        FC_outputs  = []

        for i, output in enumerate(layer_outputs):
            FC_output       = self.FC_layers[i](output)
            FC_output       = F.relu(FC_output)
            FC_outputs.append(FC_output)

        loss_pred           = self.linear(torch.cat(FC_outputs, 1))
        return loss_pred

    def reinit_model(self):
        """
        Reinitialise model parameters.
        """
        super().reinit_model()
        print("LossNet reinitialised")








    