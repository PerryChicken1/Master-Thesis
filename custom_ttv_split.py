import pandas as pd
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

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

def get_tile_proportions(data: pd.DataFrame):
    """
    Calculate proportion of observations with each tile name.

    INPUTS:
    data: output from load_data()

    OUTPUTS:
    proportions: pd.Series of tile names and proportions
    """
    vcs         = data['tile_name'].value_counts()
    proportions = vcs / vcs.sum()
    return proportions

def colour_tiles(proportions: pd.Series, prop_train: float= 0.5, prop_test: float=0.48, prop_val: float=0.02, tol: float=0.01):
    """
    Colour tiles by 'train', 'test' or 'val' with the specified proportions of observations.

    INPUTS:
    proportions: output from get_tile_proportions()
    prop_train: desired proportion of train observations
    prop_test: desired proportion of test observations
    prop_val: desired proportion of val observations
    tol: tolerance for deviation from desired proportions

    OUTPUTS:
    colouring: dictionary of tile_name : colour
    """
    # moderate inputs
    assert prop_train + prop_test + prop_val == 1, "Desired proportions must sum to 1"
    assert all(frac > 0 for frac in {prop_train, prop_test, prop_val, tol}), "Proportions and tolerance must be positive"

    # instantiate
    current_train_prop, current_test_prop, current_val_prop \
                    = 0, 0, 0
    colouring       = dict()
    remaining_tiles = proportions.copy(deep=True)
    
    # assign validation first
    while current_val_prop < (prop_val - tol):
        
        # filter to eligible tiles
        upper_limit             = prop_val + tol - current_val_prop
        eligible_tiles          = remaining_tiles[remaining_tiles < upper_limit]

        if eligible_tiles.empty:
            print(f"No eligible tiles found in validation. Current val prop is {current_val_prop}")

        # colour random tile as 'val'
        random_tile             = np.random.choice(eligible_tiles.index)
        colouring[random_tile]  = 'val'

        # recompute current val prop
        current_val_prop        += remaining_tiles[random_tile]
        remaining_tiles.drop(random_tile, inplace=True)
    
    print("Validation colouring complete")
    
    # assign test
    while current_test_prop < (prop_test - tol):
        
        # filter to eligible tiles
        upper_limit             = prop_test + tol - current_test_prop
        eligible_tiles          = remaining_tiles[remaining_tiles < upper_limit]

        if eligible_tiles.empty:
            print(f"No eligible tiles found in test. Current val prop is {current_test_prop}")

        # colour random tile as 'test'
        random_tile             = np.random.choice(eligible_tiles.index)
        colouring[random_tile]  = 'test'

        # recompute current val prop
        current_test_prop       += remaining_tiles[random_tile]
        remaining_tiles.drop(random_tile, inplace=True)

    print("Test colouring complete")

    # make remaining tiles train
    for tile in remaining_tiles.index:

        colouring[tile]         = 'train'
        current_train_prop      += remaining_tiles[tile]

    print("Train colouring complete")

    print(f"Final proportions: train = {current_train_prop}, test = {current_test_prop}, val = {current_val_prop}")
    return colouring

def plot_agbd_distributions(colouring:dict, data: pd.DataFrame):
    """
    Plot the distribution of agbd between train/ test/ validation subsets of the data.

    INPUTS:
    colouring: output from colour_tiles()
    data: output from load_data()
    """
    data['colour']  = data['tile_name'].map(colouring)
    colours         = data['colour'].unique().tolist()

    plt.figure(figsize=(10,6))
    
    for colour in colours:

        subset      = data[data['colour'] == colour]['agbd']
        subset.plot(kind='hist', density=True, alpha=0.5, label=f'{colour} tiles')

    plt.xlabel('AGBD')
    plt.ylabel('Density')
    plt.title('Density Histograms of AGBD Stratified by Tile Assignment')
    plt.legend(title='Tile assignment')

    plt.show()

def summarise_agbd_distributions(data:pd.DataFrame):
    """
    Report quintiles of AGBD in colouring.

    INPUTS:
    data: output from load_data() with 'colour' column
    """

    _, bins_train   = pd.qcut(data[data['colour'] == 'train']['agbd'], 5, retbins=True)
    _, bins_test    = pd.qcut(data[data['colour'] == 'test']['agbd'], 5, retbins=True)
    _, bins_val     = pd.qcut(data[data['colour'] == 'val']['agbd'], 5, retbins=True)

    print("Train set quintiles of agbd:")
    print(bins_train)

    print("Test set quintiles of agbd:")
    print(bins_test)

    print("Validation set quintiles of agbd:")
    print(bins_val)

def request_user_input():
    """
    Ask user whether TTV split is satisfactory.
    """
    user_input      = input("Is TTV split satisfactory? Y/N")

    while user_input not in ["Y", "N"]:

        print("Invalid input. Must be one of 'Y' or 'N' (case sensitive)")
        user_input  = input("Is TTV split satisfactory? Y/N")

    return user_input

if __name__ == "__main__":

    print("Running")
    
    data        = load_data()
    print("Loaded data!")
    
    # re-split until user is satisfied
    user_input  = "N"
    while user_input == "N":

        proportions = get_tile_proportions(data)
        print("Computed proportions!")
    
        colouring   = colour_tiles(proportions)
        print("Coloured tiles")

        plot_agbd_distributions(colouring, data)
        summarise_agbd_distributions(data)
        print("Compared agbd distributions!")

        user_input  = request_user_input()
    
    fpath   = r"C:\Users\nial\OneDrive\ETH Zürich\Master Thesis\colouring.pkl"

    with open(fpath, "wb") as output_file:
        pkl.dump(colouring, output_file)