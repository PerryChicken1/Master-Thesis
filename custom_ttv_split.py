import pandas as pd
import os
import random

def load_data():
    """
    Load features and labels, then merge into one dataset.

    RETURNS:
    data: inner join between features and labels on index
    """

    os.chdir(r"C:\Users\nial\polybox - Nial Perry (nperry@student.ethz.ch)@polybox.ethz.ch\Nial MT")
    new_features    = pd.read_csv("nial_features_2020-v1.csv")
    new_labels      = pd.read_csv("nial_labels_2020-v1.csv")

    data            = new_features.join(new_labels)
    return data

def random_colouring(n: int=33, prop_train: float=0.5, prop_test: float=0.48, prop_val: float=0.02):
    """
    Randomly 'colour' each unique tile in data. Colours are train, validation and test.
    Desired proportions are colour probabilities.

    INPUTS:
    n: number of unique tiles
    prop_train: desired train proportion
    prop_test: desired test proportion
    prop_val: desired validation proportion
    
    OUTPUTS:
    colouring: list, like ['validation', 'test', 'test', ..., 'train']
    """
    colours         = ['train', 'test', 'validation']
    probabilities   = [0.5, 0.48, 0.02]
    colouring       = []

    # ensure all colours in colouring
    while any(colour not in colouring for colour in colours):
        colouring       = random.choices(colours, weights=probabilities, k=n)

    return colouring





def calculate_agbd_quantiles(data:pd.DataFrame, train_tiles: list, val_tiles: list, test_tiles: list):
    """
    Calculate agbd quantiles given a train-test-validation split. 
    """
