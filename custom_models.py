import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam

# CUSTOM MODELS NEED fit(), predict() AND reset_model() METHODS

# MULTI-LAYER PERCEPTRON 

class MLP(nn.Module):
    
    def __init__(self, n_predictors: int=12, num_epochs: int=5, lr: float=0.01, print_freq: int=10, with_scheduler=True):

        """
        A simple three-layer perceptron model.

        INPUTS:
        n_predictors: number of predictors
        num_epochs: for training at time t
        lr: initial learning rate
        print_freq: frequency (in MABS batches) at which to print updates
        """

        super().__init__()

        self.n_predictors   = n_predictors 
        self.num_epochs     = num_epochs
        self.init_lr        = lr
        self.lr             = lr
        self.print_freq     = print_freq
        self.with_scheduler = with_scheduler
        self.fit_count      = 0

        self.layers         =   nn.Sequential(
                        nn.Linear(n_predictors, 64),
                        nn.ReLU(),
                        # nn.Linear(64, 128),
                        # nn.ReLU(),
                        # nn.Linear(128, 64),
                        # nn.ReLU(),
                        nn.Linear(64,32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.ReLU()
                        )
        
        self.path          = r"C:\Users\nial\Documents\GitHub\Master-Thesis\State dict\MLP_state_dict.pt"  
        self.save_initial_state()

    def save_initial_state(self):
        """
        Save initial parameters for later resets. 
        """
        torch.save(self.state_dict(), self.path)

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
        self.reset_model_parameters()

        # DataLoader
        X_tensor        = self.tensorize(X)
        y_tensor        = self.tensorize(y)

        tensor_dataset  = TensorDataset(X_tensor, y_tensor)
        train_loader    = DataLoader(dataset=tensor_dataset, batch_size=32, shuffle=False)

        self.train()

        optimizer       = Adam(self.parameters(), lr=self.lr)
        loss            = nn.MSELoss()

        if self.with_scheduler: 
            factor      = 0.5
            patience    = 3
            min_lr      = self.init_lr * (factor ** 3) 
            scheduler   = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)

        # train
        for epoch in range(self.num_epochs):

            epoch_loss  = 0
            n_iter      = 0

            for batch in train_loader:

                X_batch, y_batch    = batch

                optimizer.zero_grad()

                outputs             = self.forward(X_batch)
                loss_batch          = loss(outputs, y_batch)

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
        """
        self.load_state_dict(torch.load(self.path))
        
    def reset_model(self):
        """
        Reset the model parameters and attributes.
        """
        self.reset_model_parameters()
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
