#!/usr/bin/env python
import numpy as np
import random
from tqdm import tqdm
import os
import argparse
import ast
import pandas as pd

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_C_coefs")
parser.add_argument("--year", help="year of synthetic mixture", default= 2021)
parser.add_argument("--num_libs", help="number of synthtic libraries to create", default= 10)
parser.add_argument("--lib_size", help="number of synthtic libraries to create", default= 128000)
parser.add_argument("--tile", help="tile-set of the locel amodel application", default= 'X0055_Y0053')
parser.add_argument("--tree_index", help="labels of the tree species/classes in the correct order", 
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--tree_class_weights", help="labels of the tree species/classes in the correct order", 
                    default = '[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1]')
parser.add_argument("--mixture_list", help="list of mixing complexity - how many classes can be mixed in one mixture", 
                    default = '[1,2,3]' )
parser.add_argument("--mixture_weights", 
                    help="wheight for every mixing complexity - For example [1, 1, 5, 1] will increase more chances to have 3-class mixtures", 
                    default = '[1, 5, 5]' )
args = parser.parse_args()

def make_array(folder):
    files = []
    for datei in os.listdir(folder):
        files.append(os.path.join(folder,datei))
    files = sorted(files)

    array_list = []
    for file in files:
        data = np.loadtxt(file, delimiter=",")
        array_list.append(data)
    array = np.stack(array_list)
    return(array)

def mixing(year,model_number, tile=args.tile):
    #print('version ' + str(model_number))
    x_pure = make_array(os.path.join(args.working_directory, '1_pure' , f'samples_x{str(args.year)}'))
    x_pure = x_pure.astype(np.float32)
    y_pure = make_array(os.path.join(args.working_directory, '1_pure' , f'samples_y{str(args.year)}'))

    # check about nodata values
    mask = np.any(x_pure == -9999, axis=1)
    x_pure = x_pure[~mask]
    y_pure = y_pure[~mask]

    #indices_with_negatives = np.where(mask)[0]
    #print("Positionen mit no data Werten:", indices_with_negatives)

    #------------------------------------------------------------
    # calculate probabilities for sampling based on the tile's point list
    #------------------------------------------------------------
    probabilities = pd.read_csv('/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tiles_point_lists/'+ tile +'.txt',
                                sep=',', engine='python')
    sets = probabilities['Set'].values
    sets = sets[~mask]
    probs = np.zeros(len(y_pure))
    for i in args.tree_index.strip('[]').split(','):
        mask = (y_pure == int(i)) & (sets == 1)
        if len(y_pure[mask]) > 0:
            probs[mask] = 1 / len(y_pure[mask])
        mask = (y_pure == int(i)) & (sets == 0)
        if len(y_pure[mask]) > 0:
            probs[mask] = 0

    #------------------------------------------------------------
    #                   perform the mixing
    #------------------------------------------------------------
    training_sample = int(args.lib_size)
    x_mixed = []
    y_mixed = []

    lc_index = ast.literal_eval(args.tree_index)
    list_of_indices = []

    for _ in range(training_sample):
        k = random.choices(ast.literal_eval(args.mixture_list), 
                           weights= ast.literal_eval(args.mixture_weights), k=1 )[0]
        fractions = np.random.dirichlet(np.ones(k),size=1)[0]
        chosen_classes = random.choices(lc_index, k=k, weights= ast.literal_eval(args.tree_class_weights))

        selected_indices = []
        # select index for each class
        for cls in chosen_classes:
            indices = np.where(y_pure == cls)[0]
            selected_idx = random.choices(indices, weights=probs[indices]) # weights are the probabilities calculated based on the tile's point list
            selected_indices.append(selected_idx[0])
            list_of_indices.append(selected_idx[0])
        chosen_index = selected_indices
        
        x = 0
        y = np.zeros(len(lc_index))
        for i in range(len(chosen_index)):
            x += x_pure[chosen_index[i]]*fractions[i]
            label_pos = lc_index.index(y_pure[chosen_index[i]])
            y[label_pos] += fractions[i]
        x_mixed.append(x)
        y_mixed.append(y)
    x_mixed = np.array(x_mixed, np.int16)
    y_mixed = np.array(y_mixed, np.float32)

    y_mixed = np.multiply(y_mixed, 100)
    y_mixed_int = y_mixed.astype(np.int16)

    #------------------------------------------------------------
    #                   store the mixed data
    #------------------------------------------------------------
    if not os.path.exists(os.path.join(args.working_directory, '2_mixed_data_tile', tile ,'version' +str(model_number))):
        os.makedirs(os.path.join(args.working_directory, '2_mixed_data_tile', tile , 'version' +str(model_number)))
    #x_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_tile', tile ,'version' +str(model_number), 'x_mixed_' + str(year) + '.npy')
    #y_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_tile', tile ,'version' +str(model_number), 'y_mixed_' + str(year) + '.npy')
    #np.save(x_mixed_out_path, arr=x_mixed)
    #np.save(y_mixed_out_path, arr=y_mixed)
    x_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_tile', tile ,'version' +str(model_number), 'x_mixed_' + str(year) + '.npz')
    y_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data_tile', tile ,'version' +str(model_number), 'y_mixed_' + str(year) + '.npz')
    np.savez_compressed(x_mixed_out_path, x_mixed=x_mixed)
    np.savez_compressed(y_mixed_out_path, y_mixed=y_mixed_int)
    # load example: data = np.load(x_mixed_out_path); x_mixed_loaded = data['x_mixed']


if __name__ == '__main__':
    # only for testing
    #mixing(args.year,1, args.tile)
    #-----------------------------------

    #-------- parallel one tile --------
    #Parallel(n_jobs=10, backend="loky")(delayed(mixing)(args.year,i+1, args.tile) for i in range(int(args.num_libs)))
    #-----------------------------------
    
    #------- Only One Federal state (RLP) ------
    num_workers = 50
    # germany: 
    #with open("/data/ahsoka/eocp/forestpulse/02_scripts/spline/ProcessingPlan/germany_tiles_sortY.txt",'r') as file:
    with open('/data/ahsoka/eocp/forestpulse/02_scripts/spline/fed_states_tiles/RLP_tiles_allY.txt', 'r') as file:
        # Read all lines into a list
        lines = file.readlines()
    list_tiles = [line.strip() for line in lines]
    list_tiles = list_tiles[1:] # remove header
    Parallel(n_jobs=num_workers, backend="loky")(
        delayed(mixing)(args.year,i+1, tile) for i in range(int(args.num_libs)) for tile in tqdm(list_tiles))
    #-----------------------------------
