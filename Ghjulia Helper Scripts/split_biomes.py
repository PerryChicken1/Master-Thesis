"""
TODO
"""

# Broad LC classes definitions

classes = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126, 200, 255]
classes_to_ignore = [0, 50, 70, 80, 200, 255]
classes_to_include = [x for x in classes if x not in classes_to_ignore]
descriptions = ['Unknown', 'Shrubs', 'Herbaceous vegetation', 'Cultivated and managed vegetation / agriculture', 'Urban / built up', 'Bare / sparse vegetation', 'Snow and ice', 'Permanent water bodies', 'Herbaceous wetland', 'Moss and lichen', 'Closed forest, evergreen needle leaf', 'Closed forest, evergreen broad leaf', 'Closed forest, deciduous needle leaf', 'Closed forest, deciduous broad leaf', 'Closed forest, mixed', 'Closed forest, other', 'Open forest, evergreen needle leaf', 'Open forest, evergreen broad leaf', 'Open forest, deciduous needle leaf', 'Open forest, deciduous broad leaf', 'Open forest, mixed', 'Open forest, other', 'Oceans, seas', 'Masked']
colors = ['#282828', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4', '#f0f0f0', '#0032c8', '#0096a0', '#fae6a0', '#58481f', '#009900', '#70663e', '#00cc00', '#4e751f', '#007800', '#666000', '#8db400', '#8d7400', '#a0dc00', '#929900', '#648c00', '#000080', 'white']

############################################################################################################################
# IMPORTS

import rasterio as rs
from os.path import join
import glob
import tqdm
import pickle
import numpy as np
import itertools
from scipy.special import comb
from iteration_utilities import random_combination
import argparse
from os.path import exists

############################################################################################################################
# Helper functions

def setup_parser() :
    """ 
    Setup the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Paths arguments
    parser.add_argument('--path_txt', type = str, help = 'Path to the folder containing the .txt files listing the S2 tiles.',
                        default = '/scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel')
    parser.add_argument('--path_lc', type = str, help = 'Path to the folder containing the land cover tiles.',
                        default = '/scratch2/gsialelli/LC')
    parser.add_argument('--path_output', type = str, help = 'Path to the folder where the output files will be saved.',
                        default = '/scratch2/gsialelli/BiomassDatasetCreation/Data/download_Sentinel/biomes_split')
    parser.add_argument('--AOIs', type = str, nargs = '+', help = 'List of AOIs to process.', required = True)
    parser.add_argument('--year', type = int, help = 'Year of the land cover tiles.', required = True)
    parser.add_argument('--threshold', type = float, help = 'Threshold for the similarity between the distributions.', 
                        default = 0.05)
    parser.add_argument('--max_iter', type = int, help = 'Maximum number of iterations to try to find a valid split.', 
                        default = 10000)

    # Parse the arguments
    args = parser.parse_args()
    
    return args.AOIs, args.path_txt, args.path_lc, args.path_output, args.year, args.threshold, args.max_iter


def per_tile_dist(path_lc, year = 2019, _save = False) :
    """
    Compute the histogram frequencies.

    Args:
    - path_lc (str): path to the folder containing the tiles
    - year (int): year of the tiles

    Returns:
    - dict: a dictionary containing the distributions of the biomes for each tile
    """
    
    tilenames = glob.glob(join(path_lc, f'LC_*_{year}.tif'))

    if exists(join(path_lc, f'distributions_{year}.pkl')) : 
        print('Loading existing distributions...')
        with open(join(path_lc, f'distributions_{year}.pkl'), 'rb') as f :
            distributions = pickle.load(f)
        already_tilenames = list(distributions.keys())
        print(f'Found {len(already_tilenames)} tiles already processed...')
        tilenames = [t for t in tilenames if t.split('_')[-2] not in already_tilenames]
        print(f'Found {len(tilenames)} new tiles to process...')

    else: distributions = {}
    
    for tile_path in tqdm.tqdm(tilenames) :

        tile = tile_path.split('_')[-2]
        distributions[tile] = []
        
        with rs.open(tile_path, 'r') as src :
            tile_data = src.read(1)
        
        for biome in classes_to_include :
            biome_data = tile_data[tile_data == biome]
            distributions[tile].append(len(biome_data))
    
    if _save == True :
        with open(join(path_lc, f'distributions_{year}.pkl'), 'wb') as f :
            pickle.dump(distributions, f)

    return distributions


def compute_sim(ref_dist, test_dist, threshold) :
    """
    Compute the similarity between two distributions. This is done as follows: we compute the absolute difference
    between the two distributions, and if the difference for any of the biomes is greater than a certain threshold,
    we consider the distributions to be too different. 

    Args:
    - ref_dist (np.array): the reference distribution, where the sum of all values is 1
    - test_dist (np.array): the test distribution, where the sum of all values is 1

    Returns:
    - bool: whether the distributions are similar or not
    - np.array: the differences between the two distributions
    """

    diff = np.abs(np.subtract(ref_dist, test_dist))
    if np.any(diff > threshold) :
        valid = False
    else: valid = True
    
    return valid, diff


def generate_splits(AOIs, path_txt, path_lc, year, max_iter, threshold = 0.05, min_pct = 0.001) :
    """
    
    Iterating over all existing combinations is impossible, so we randomly generate combinations.
    
    
    """

    all_splits = {}

    for AOI in AOIs :

        print('Processing AOI:', AOI)

        all_splits[AOI] = []
        
        # Load the list of tiles for the AOI
        with open(join(path_txt, f'Sentinel_Clem_{AOI}.txt'), 'r') as f :
            tiles = np.array([t.strip() for t in f.readlines()])
        
        # Load the pre-computed per-tile distributions
        with open(join(path_lc, f'distributions_{year}.pkl'), 'rb') as f :
            distributions = pickle.load(f)
        
        # Get the distribution for all tiles combined
        tiles_distributions = [distributions[tile] for tile in tiles]
        aoi_distribution = np.add.reduce(tiles_distributions)
        aoi_distribution = aoi_distribution / np.sum(aoi_distribution)

        # Get which biomes are represented across this AOI
        biomes_mask = (aoi_distribution >= min_pct)

        # Now we want to split the tiles into 3 groups
        # Split the tiles into 3 groups:
        # - train: 65% of the data
        # - val: 15% of the data
        # - test: 20% of the data

        # Compute the length of each group
        num_tiles = len(tiles)
        len_train = int(num_tiles * 0.65)
        len_val = int(num_tiles * 0.15)
        
        # Randomly sample train combinations
        for _ in range(max_iter):

            train_comb = random_combination(tiles, r = len_train)

            # Check if the distribution of the train tiles is similar to the distribution of the AOI
            train_distributions = np.add.reduce([distributions[tile] for tile in train_comb])
            train_distributions = train_distributions / np.sum(train_distributions)

            # Check that all required biomes are represented
            if np.any(train_distributions[biomes_mask] == 0) :
                continue

            valid_train, similarity_train = compute_sim(aoi_distribution, train_distributions, threshold)
            if valid_train :

                # Remove the train tiles from the list of tiles
                comb_tiles = tiles[np.isin(tiles, train_comb, invert = True, assume_unique = True)]
                
                # Randomly sample validation combinations
                for _ in range(max_iter):

                    val_comb = random_combination(comb_tiles, r = len_val)

                    # Check if the distribution of the val tiles is similar to the distribution of the AOI
                    val_distributions = np.add.reduce([distributions[tile] for tile in val_comb])
                    val_distributions = val_distributions / np.sum(val_distributions)


                    # Check that all required biomes are represented
                    if np.any(val_distributions[biomes_mask] == 0) :
                        continue
                
                    valid_val, similarity_val = compute_sim(aoi_distribution, val_distributions, threshold)
                    if valid_val :

                        # Remove the val tiles from the list of tiles (and cast to tuple to be aligned with the others)
                        test_tiles = tuple(comb_tiles[np.isin(comb_tiles, val_comb, invert = True, assume_unique = True)])
                        
                        test_distributions = np.add.reduce([distributions[tile] for tile in test_tiles])
                        test_distributions = test_distributions / np.sum(test_distributions)

                        # Check that all required biomes are represented
                        if np.any(test_distributions[biomes_mask] == 0) :
                            continue

                        valid_test, similarity_test = compute_sim(aoi_distribution, test_distributions, threshold)
                        if valid_test :

                            # Store the combination and the similarity scores
                            all_splits[AOI].append({'train': train_comb, 'val': val_comb, 'test': test_tiles, 'similarity': {'train': similarity_train, 'val': similarity_val, 'test': similarity_test}})

                        else: continue

                    else: continue 
            
            else: continue
    
    return all_splits


############################################################################################################################
# Execute

if __name__ == '__main__' :

    AOIs, path_txt, path_lc, path_output, year, threshold, max_iter = setup_parser()
    
    with np.printoptions(formatter={'float':           "{0:0.3f}".format}, linewidth=np.inf) :
        
        # Get some valid splits for all of the AOIs
        all_splits = generate_splits(AOIs, path_txt, path_lc, year, max_iter, threshold = threshold)

        # Now get the best split for each AOI
        final_splits = {}
        
        for aoi in AOIs:
            
            aoi_res = all_splits[aoi]
            if aoi_res == [] :
                print(f'No valid split found for AOI {aoi}!')
                AOIs.remove(aoi)
                continue

            scores = []
            for comb in aoi_res :
                score = sum([np.sum(comb['similarity'][mode]) for mode in ['train','val','test']])
                scores.append(score)

            # Get the index in scores with the lowest value
            idx = np.argmin(scores)
            final_splits[aoi] = aoi_res[idx]
        
        suffix = '-'.join(AOIs)
        with open(join(path_output, f'final_splits_{year}_{suffix}.pkl'), 'wb') as f :
            pickle.dump(final_splits, f)


"""
Limitations :
. Would benefit from adding a constraint that neighbouring tiles should not be in the same split as much as possible
. Split is based on the assumption that all tiles carry the same amount of information, which is not the case (e.g. tiles mostly over water). Could assign a weight
  to each tile, and sample based on that weight to reach the required percentage. 
"""