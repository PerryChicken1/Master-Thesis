
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam

# CUSTOM MODELS NEED fit(), predict() AND reset_model() METHODS

# MULTI-LAYER PERCEPTRON 

class MLP(nn.Module):
    
    def __init__(self, n_predictors: int=12, num_epochs: int=5, lr: float=0.01, print_freq: int=100):

        """
        A simple three-layer perceptron model.

        INPUTS:
        n_predictors: number of predictors
        num_epochs: for training at time t
        lr: initial learning rate
        """

        super().__init__()

        self.n_predictors   = n_predictors 
        self.num_epochs     = num_epochs
        self.init_lr        = lr
        self.lr             = lr
        self.print_freq     = print_freq
        self.t              = 0
                
        self.loss           = nn.MSELoss()

        self.layers         =   nn.Sequential(
                        nn.Linear(n_predictors, 64),
                        nn.ReLU(),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.ReLU()
                        )
        
    def forward(self, x):
        output  = self.layers(x)
        return torch.flatten(output)
    
    @staticmethod
    def tensorize(array:np.ndarray):
        """
        Convert numpy nd array into a tensor.

        INPUTS:
        array: nd array

        OUTPUTS:
        tensor: tensor with float32s
        """
        return torch.tensor(array, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run training loop at time t.

        INPUTS:
        X: data
        y: labels
        """
        # clean slate
        # self.reset_model_parameters()

        # DataLoader
        X_tensor        = self.tensorize(X)
        y_tensor        = self.tensorize(y)

        tensor_dataset  = TensorDataset(X_tensor, y_tensor)
        train_loader    = DataLoader(dataset=tensor_dataset, batch_size=32, shuffle=False)

        self.train()

        optimizer       = Adam(self.parameters(), lr=self.lr)
        scheduler       = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        # train
        for epoch in range(self.num_epochs):

            epoch_loss  = 0
            n_iter      = 0

            for batch in train_loader:

                X_batch, y_batch    = batch

                optimizer.zero_grad()

                outputs             = self.forward(X_batch)
                loss_batch          = self.loss(outputs, y_batch)

                loss_batch.backward()
                optimizer.step()

                epoch_loss          += loss_batch.item()
                n_iter              += 1

            epoch_loss /= n_iter
            scheduler.step(epoch_loss)
            self.lr     = scheduler.get_last_lr() # track lr

            # intermittent updates
            if self.t % self.print_freq == 0: 
                print(f'Timestep {self.t}. Epoch [{epoch+1}/{self.num_epochs}], Training loss: {epoch_loss:.4f}, Current learning rate: {self.lr}')
            
        self.t += 1

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
        """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def reset_model(self):
        """
        Reset the model parameters and attributes.
        """
        self.reset_model_parameters()
        self.t          = 0
        self.lr         = self.init_lr
        print(f"MLP model reset")

