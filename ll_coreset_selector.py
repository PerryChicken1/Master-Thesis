
import numpy as np
import torch
import random
from warnings import warn
from typing import Set, Optional, Self
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn import functional as F
from custom_models import *

# CORESET SELECTOR COMBINES LOSSNET AND MLP TO SELECT A CORESET
class LL_coreset_selector:
    
    def __init__(self, n_predictors: int=12, num_epochs: int=5, lr_MLP: float=0.01, lr_LL:float=1e-3\
                 , print_freq: int=10, with_scheduler=True, loss_fn=nn.MSELoss, FC_dim: int=32):

        """
        Coreset selector with loss learning.

        Args:
            n_predictors: number of predictors
            num_epochs: for training at time t
            lr_MLP: initial learning rate for MLP
            lr_LL: initial learning rate for LossNet
            print_freq: frequency (in MABS batches) at which to print updates
            with_scheduler: use learning rate scheduler?
            loss_fn: loss function to train model
            FC_dim: output size of fully-connected layers (embeddings -> concat)
        """

        self.__dict__.update(vars())
        
        self.init_lr_MLP    = lr_MLP
        self.init_lr_LL     = lr_LL
        self.fit_count      = 0
        self.ll_loss_weight = 8000 / 15 # λ in Yoo and Kweon (2019) -> chosen to bring losses to equal scale

        self.MLP            = MLP_loss_learn(n_predictors, num_epochs, lr_MLP, print_freq, with_scheduler, loss_fn)
        self.loss_net       = self.init_LossNet(FC_dim)

    def update_MLP_architecture(self, new_architecture: nn.Sequential):
        """
        Update architecture of MLP.

        Args:
            new_architecture (nn.Sequential): new MLP architecture.
        """
        self.MLP.update_architecture(new_architecture)
        self.loss_net       = self.init_LossNet(self.FC_dim)

    @staticmethod
    def squared_error_loss(input: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        Return a vector of squared errors as the observation-wise losses.

        Args:
            input (torch.tensor): predicted `agbd` for one batch
            target (torch.tensor): true `agbd` for one batch

        Returns:
            loss (torch.tensor): observation-wise losses
        """
        sq_loss = (input - target) ** 2
        return sq_loss

    def init_LossNet(self, FC_dim: int) -> nn.Module:
        """
        Initialise LossNet.
        
        Args:
            FC_dim (int): output size for layers
        
        Returns:
            loss_net (nn.Module): LossNet neural network
        """
        # infer embedding sizes from self.layers
        embedding_sizes = []
        
        for layer in self.MLP.layers:
            if isinstance(layer, nn.Linear):
                embedding_size  = layer.out_features
                embedding_sizes.append(embedding_size)

        # remove output layer
        loss_net        = LossNet(embedding_sizes[:-1], FC_dim)
        return loss_net
    
    @staticmethod
    def compute_ll_loss(input: torch.tensor, target: torch.tensor, margin=1.0) -> torch.tensor:
        """
        Equation (2) from Yoo and Kweon (2019).

        Args:
            input (torch.tensor): predicted losses for one batch
            target (torch.tensor): true losses for one batch
            margin (float): constant determining the scale of loss

        Returns:
            loss (torch.tensor): computed loss
        """
        assert len(input) % 2 == 0, 'The batch size is not even!'
        assert input.shape == input.flip(0).shape, 'Tensor shape is off...'

        input   = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target  = (target - target.flip(0))[:len(target)//2]
        target  = target.detach()

        one     = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
        loss    = torch.sum(torch.clamp(margin - one * input, min=0))
        loss    /= input.size(0) # Note that the size of input is already halved

        return loss

    def fit(self, X: np.ndarray, y:np.ndarray) -> Self:
        """
        Jointly train MLP and lossnet using inputted data.

        Args:
            X: data
            y: targets
        """

        # reset network parameters
        self.MLP.reset_model_parameters()
        self.loss_net.reset_model_parameters()

        X_tensor            = MLP.tensorize(X)
        y_tensor            = MLP.tensorize(y)

        tensor_dataset      = TensorDataset(X_tensor, y_tensor)
        train_loader        = DataLoader(dataset=tensor_dataset, batch_size=32, shuffle=True)

        self.MLP.train()
        self.loss_net.train()

        optimiser_MLP       = Adam(self.MLP.parameters(), lr=self.lr_MLP)
        optimiser_loss_net  = Adam(self.loss_net.parameters(), lr=self.lr_LL)

        loss_fn             = self.squared_error_loss

        if self.with_scheduler:
            raise NotImplementedError("`fit()` method currently does not handle learning rate schedulers") 
            factor          = 0.5
            patience        = 3
            min_lr          = self.init_lr_MLP * (factor ** 4) 
            scheduler       = lr_scheduler.ReduceLROnPlateau(optimiser_MLP, mode='min', factor=factor, patience=patience, min_lr=min_lr)
                    
        self.fit_count += 1

        for epoch in range(self.num_epochs):
            
            loss_combined_epoch = 0
            loss_MLP_epoch      = 0
            loss_LL_epoch       = 0
            n_iter              = 0

            for batch in train_loader:
                
                optimiser_MLP.zero_grad()
                optimiser_loss_net.zero_grad()

                X_batch, y_batch    = batch

                # ensure even batch size
                batch_size          = len(y_batch)
                if batch_size % 2   == 1: continue

                # MLP outputs 
                outputs, embeddings\
                                    = self.MLP.forward(X_batch)
                
                # MLP losses
                loss_batch_MLP      = loss_fn(y_batch, outputs) # to train LossNet
                loss_MLP            = loss_batch_MLP.mean() # for .backward() pass

                # predicted losses
                pred_losses         = self.loss_net.forward(embeddings)
                self.pred_losses    = pred_losses
                self.loss_batch_MLP = loss_batch_MLP

                # LossNet loss (already averaged by `compute_ll_loss()`)
                loss_LL             = self.compute_ll_loss(pred_losses, loss_batch_MLP)

                # combined loss
                loss_combined       = loss_MLP + self.ll_loss_weight * loss_LL    

                # update parameters and record losses
                loss_combined.backward()
                optimiser_MLP.step()
                optimiser_loss_net.step()

                loss_combined_epoch += loss_combined.item()
                loss_MLP_epoch      += loss_MLP.item()
                loss_LL_epoch       += loss_LL.item()
                n_iter              += 1

            loss_combined_epoch /= n_iter
            loss_MLP_epoch      /= n_iter
            loss_LL_epoch       /= n_iter

            if self.with_scheduler: 
                raise NotImplementedError("Scheduler cannot be handled!")
                scheduler.step(loss_combined_epoch)
                self.lr         = scheduler.get_last_lr()[0] # track lr

            # intermittent updates
            if self.fit_count % self.print_freq == 1: 
                print(f''' Model fit {self.fit_count}. Epoch [{epoch+1}/{self.num_epochs}].
                      Epoch losses: combined = {loss_combined_epoch:.4f}, MLP = {loss_MLP_epoch:.4f}, LL = {loss_LL_epoch:.4f}.
                      MLP lr: {self.lr_MLP}, LL lr: {self.lr_LL}''')


        return self
 
    def predict_losses(self, X: np.ndarray) -> torch.Tensor:
        """
        Predict losses for MLP on data X.

        Args:
            X: input data

        Returns:
            pred_losses: `LossNet` estimate of `MLP` prediction loss
        """
        self.loss_net.eval()
        self.MLP.eval()

        X                       = MLP.tensorize(X)

        with torch.no_grad():
            _, embeddings       = self.MLP.forward(X)
            pred_losses         = self.loss_net.forward(embeddings)
            return pred_losses
        
    @staticmethod
    def top_K_losses(pred_losses, K: int, indices: list) -> torch.Tensor:
        """
        Get indices of top K predicted losses.

        Args:
            pred_losses: output from `predict_losses()`
            K: number of largest losses to return
            indices: indices from which top K selected (i.e., converting positional -> absolute indexing)

        Returns: 
            top_K_idx: top K indices
        """
        _, top_K_rel_idx    = torch.topk(pred_losses, K, dim=0)
        top_K_idx           = [indices[i] for i in top_K_rel_idx]
        return top_K_idx
