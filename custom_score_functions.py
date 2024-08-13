import numpy as np
import torch
from torch import nn

# LOG RATIO SCORE COMPATIBLE WITH TRAINING AGENT
def log_ratio(y: np.ndarray, y_hat: np.ndarray):
    """
    The log ratio is a symmetric and relative accuracy score (Tofallis, 2015)
    """
    # compute the log quotient
    log_quotient    = np.log(y_hat/y)

    # replace inf values by large penalty (if y_hat == 0)
    log_quotient    = np.where(np.isinf(log_quotient), 10.0, log_quotient)

    # replace nan values by large penalty (if y == 0)
    log_quotient    = np.where(np.isinf(log_quotient), 10.0, log_quotient)

    # mean squared residual
    mean_error      = np.mean(np.square(log_quotient))

    # return error
    return mean_error

def LogRatioLoss(y: torch.tensor, y_hat: torch.tensor):
    """
    The log ratio is a symmetric and relative accuracy score (Tofallis, 2015)
    """       
    # compute the log quotient
    log_quotient    = torch.log(y_hat / y)

    # replace inf values by large penalty (if y_hat == 0)
    log_quotient    = torch.where(torch.isinf(log_quotient), torch.tensor(10.0), log_quotient)

    # replace nan values by large penalty (if y == 0)
    log_quotient    = torch.where(torch.isnan(log_quotient), torch.tensor(10.0), log_quotient)
        
    # Mean squared residual
    mean_error      = torch.mean(torch.square(log_quotient))
        
    return mean_error

# LOG RATIO LOSS FOR TRAINING NN
# class LogRatioLoss(nn.Module):
    def __init__(self):
        super(LogRatioLoss, self).__init__()
    
    def forward(self, y: torch.tensor, y_hat: torch.tensor):
        
        # compute the log quotient
        log_quotient    = torch.log(y_hat / y)
        
        # Mean squared residual
        mean_error      = torch.mean(torch.square(log_quotient))
        
        return mean_error
