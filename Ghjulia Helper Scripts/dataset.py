"""

This script is used to extract the features and labels of the central pixels from the h5 files and save them as csv files.

"""

############################################################################################################################
# IMPORTS

from nial_dataset import *
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

############################################################################################################################
# Helper functions

def construct_feature_names(features_selection) :

    feature_names = []
    feature_names += features_selection['bands']
    if features_selection['latlon'] : feature_names += ['lat_cos', 'lat_sin', 'lon_cos', 'lon_sin']
    else: feature_names += ['lat_cos', 'lat_sin']
    if features_selection['gedi_dates'] : feature_names += ['gedi_num_days', 'gedi_doy_cos', 'gedi_doy_sin']
    if features_selection['gedi_all'] : feature_names += ['agbd_se', 'elev_lowes', 'pft_class', 'region_cla', 'rh98', 'selected_a', 'sensitivit', 'solar_elev', 'urban_prop']
    if features_selection['alos'] : feature_names += ['alos_hh', 'alos_hv']
    if features_selection['ch'] : feature_names += ['ch', 'ch_std']
    if features_selection['lc'] : feature_names += ['lc_cos', 'lc_sin', 'lc_prob']
    if features_selection['dem'] : feature_names += ['dem']
    
    return feature_names


def create_table(paths, mode, year = 2019, version = 4) :
    """
    This function creates the features and labels csv files for the train, val and test sets.
    You can run it locally with the following arguments:
        - fnames = ['data_subset-2020-v4_6-20.h5', 'data_subset-2020-v4_7-20.h5', 'data_subset-2020-v4_8-20.h5']
        - paths = {'h5':'/scratch2/gsialelli/patches', 'norm': '/scratch2/gsialelli/patches', 'map': '/scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split'}

    Args:
    - fnames : list : the names of the h5 files

    Returns:
    - None
    """

    # Set up arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.latlon = True
    args.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    args.ch = True
    args.s1 = False
    args.alos = True
    args.lc = True
    args.dem = True
    args.patch_size = [1,1]
    args.norm_strat = 'pct'
    args.norm = False
    args.s2_dates = False
    args.gedi_dates = True
    args.gedi_all = True

    # Get the feature names
    features_names = construct_feature_names(args)

    # Iterate over the modes
    print(f'Processing {mode} data...')

    # Get the dataset
    custom_dataset = GEDIDataset(paths, years = [year], chunk_size = 1, mode = mode, args = args)
    data_loader = DataLoader(dataset = custom_dataset,
                            batch_size = 1024,
                            shuffle = False,
                            num_workers = 8)

    # Iterate through the DataLoader
    print('starting to iterate...')
    
    for batch_idx, batch_samples in enumerate(tqdm(data_loader)):
        
        features, labels = batch_samples
        features = features.squeeze(2).squeeze(2).numpy().astype(np.float32)
        labels = labels.numpy().astype(np.float32)

        assert features.shape[1] == len(features_names), f'Expected {len(features_names)} features, got {features.shape[1]}'

        if batch_idx == 0 :
            df_features = pd.DataFrame(features, columns = features_names)
            df_labels = pd.DataFrame(labels, columns = ['agbd'])
        else: 
            df_features = pd.concat([df_features, pd.DataFrame(features, columns=features_names)], ignore_index=True)
            df_labels = pd.concat([df_labels, pd.DataFrame(labels, columns=['agbd'])], ignore_index=True)
        
    print('done!')
    print()

    # Save the data
    df_features.to_csv(join(paths['h5'], f'{mode}_features_{year}-v{version}.csv'), index = False)
    df_labels.to_csv(join(paths['h5'], f'{mode}_labels_{year}-v{version}.csv'), index = False)
    


class RF_GEDIDataset(Dataset) :
    """
    This class is a subclass of torch.utils.data.Dataset. It is used to load the features and labels from the csv files.
    """

    def __init__(self, data_path, mode, features) :
        """
        This function initializes the class.

        Args:
        - data_path: str, path to the .csv files
        - mode: str, the mode of the dataset (i.e. train, test, or val)
        - features: dict, selection of the input features to load

        Returns:
        - None
        """

        # Get the features to be used
        columns_to_load = construct_feature_names(features)
        if mode == 'train' : print('Loading features:', columns_to_load)

        # Load the features
        self.features = pd.read_csv(join(data_path, f'{mode}_features_2020-vghana.csv'), usecols = columns_to_load)
        
        # Load the labels
        self.labels = pd.read_csv(join(data_path, f'{mode}_labels_2020-vghana.csv'))


"""
Example:

data_path = '/scratch2/gsialelli/patches'
mode = 'test'
features = {'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'], 'latlon': True, 'alos': True, 'ch': True, 'lc': True, 'dem': True, 'gedi_dates': True, 'gedi_all': True}
train_dataset = RF_GEDIDataset(data_path, mode = mode, features = features)
"""