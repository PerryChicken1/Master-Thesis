import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn import functional as F
from custom_score_functions import LogRatioLoss, NUGGET

# CUSTOM MODELS NEED fit(), predict() AND reset_model() METHODS

# MULTI-LAYER PERCEPTRON 

class MLP(nn.Module):
    
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

        super().__init__()

        self.n_predictors   = n_predictors 
        self.num_epochs     = num_epochs
        self.init_lr        = lr
        self.lr             = lr
        self.print_freq     = print_freq
        self.with_scheduler = with_scheduler
        self.loss_fn        = loss_fn
        self.fit_count      = 0

        self.layers         = self.build_model()
        
        self.path          = r"C:\Users\nial\Documents\GitHub\Master-Thesis\State dict\MLP_state_dict.pt"  
        self.save_initial_state()

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
    
    def update_architecture(self, new_architecture: nn.Sequential):
        """
        Update MLP architecture of self.

        Args:
            new_architecture: Sequential object of layers and activations
        """
        self.layers = new_architecture
        self.reset_model()
        self.save_initial_state()
    
    def save_initial_state(self):
        """
        Save initial parameters for later resets. 
        """
        torch.save(self.state_dict(), self.path)

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

        optimizer       = Adam(self.parameters(), lr=self.lr)
        loss_fn         = self.loss_fn

        if self.with_scheduler: 
            factor      = 0.5
            patience    = 3
            min_lr      = self.init_lr * (factor ** 4) 
            scheduler   = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)

        # train
        for epoch in range(self.num_epochs):
        
            epoch_loss  = 0
            n_iter      = 0

            for batch in train_loader:

                X_batch, y_batch    = batch

                optimizer.zero_grad()

                outputs             = self.forward(X_batch)
                
                loss_batch          = loss_fn(y_batch, outputs)

                loss_batch.backward()
                optimizer.step()

                epoch_loss          += loss_batch.item()
                n_iter              += 1

            epoch_loss /= n_iter

            if self.with_scheduler: 
                scheduler.step(epoch_loss)
                self.lr     = scheduler.get_last_lr()[0] # track lr

            # intermittent updates
            if self.fit_count % self.print_freq == 0: 
                print(f'Batch {self.fit_count}. Epoch [{epoch+1}/{self.num_epochs}], Training loss: {epoch_loss:.4f}, lr: {self.lr}')
            
        self.fit_count += 1

    def predict(self, X: np.ndarray):
        """
        Predict with current model.
        """
        self.eval()

        X_tensor    = self.tensorize(X)

        with torch.no_grad():
            return self.forward(X_tensor)
        
    def reset_model_parameters(self):
        """
        Reset the model parameters.

        Args:
            reset_model: whether to reinitialise parameters in the layers
        """
        self.load_state_dict(torch.load(self.path))
        
    def reset_model(self):
        """
        Reset the model parameters and attributes.
        """
        def reset_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.apply(reset_weights)
        print("First 10 weights in layer 0: \n", self.state_dict()['layers.0.weight'][0][0:10])

        self.fit_count  = 0
        self.lr         = self.init_lr
        print(f"MLP model reset")

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
                 , print_freq: int=10, with_scheduler=True, loss_fn=nn.MSELoss, FC_dim: int=32):

        """
        Perceptron model extended to acommodate jointly learning the loss.

        Source: https://arxiv.org/pdf/1905.03677

        Args:
            n_predictors: number of predictors
            num_epochs: for training at time t
            lr: initial learning rate
            print_freq: frequency (in MABS batches) at which to print updates
            with_scheduler: use learning rate scheduler?
            loss_fn: loss function to train model
            FC_dim: output size of fully-connected layers (embeddings -> concat)
        """ 
        super().__init__(n_predictors, num_epochs, lr, print_freq, with_scheduler, loss_fn)

        self.loss_net    = self.init_LossNet(FC_dim)

    def init_LossNet(self, FC_dim: int) -> nn.Module:
        """
        Initialise LossNet.
        
        Args:
            embedding_sizes: input sizes for layers
            FC_dim: output size for layers
        """
        # infer embedding sizes from self.layers
        embedding_sizes = []
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                embedding_size  = layer.out_features

        loss_net        = LossNet(embedding_sizes[:-1], FC_dim)
        return loss_net

    def forward(self, x):

        output          = x
        layer_outputs   = []

        for i, layer in enumerate(self.layers):
            output = layer(output)

            # store results from activation layers
            if i % 2 == 1:
                layer_outputs[i] = torch.flatten(output)

        return torch.flatten(output), layer_outputs[:-1]

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

        optimizer       = Adam(self.parameters(), lr=self.lr)
        loss_fn         = self.loss_fn

        if self.with_scheduler: 
            factor      = 0.5
            patience    = 3
            min_lr      = self.init_lr * (factor ** 4) 
            scheduler   = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)

        # train
        for epoch in range(self.num_epochs):
        
            epoch_loss  = 0
            n_iter      = 0

            for batch in train_loader:

                X_batch, y_batch    = batch

                optimizer.zero_grad()

                outputs             = self.forward(X_batch)
                
                loss_batch          = loss_fn(y_batch, outputs)

                loss_batch.backward()
                optimizer.step()

                epoch_loss          += loss_batch.item()
                n_iter              += 1

            epoch_loss /= n_iter

            if self.with_scheduler: 
                scheduler.step(epoch_loss)
                self.lr     = scheduler.get_last_lr()[0] # track lr

            # intermittent updates
            if self.fit_count % self.print_freq == 0: 
                print(f'Batch {self.fit_count}. Epoch [{epoch+1}/{self.num_epochs}], Training loss: {epoch_loss:.4f}, lr: {self.lr}')
            
        self.fit_count += 1

class LossNet(nn.Module):
    
    def __init__(self, embedding_sizes: list = [64, 32], FC_dim: int = 32):
        """
        Network for predicting MLP losses based on MLP embeddings.

        Args:
            n_layers: number of parameterised layers in MLP
            FC_dim: number of nodes in fully-connected layers 
        """

        super().__init__()

        self.n_layers   = len(embedding_sizes)
        self.FC_dim     = FC_dim

        # FC layers (embeddings -> concatenation)
        self.FC_layers  = self._init_FC_layers(embedding_sizes, FC_dim)

        # final layer (concatenation -> loss prediction)
        self.linear     = nn.Linear(self.n_layers * FC_dim, 1)
    
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
            FC_outputs[i]   = FC_output

        loss_pred           = self.linear(torch.cat(FC_outputs, 1))
        return loss_pred







    