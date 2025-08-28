import numpy as np
import random
from tqdm import tqdm
import os
import argparse
import ast
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2022")
parser.add_argument("--year", help="year of synthetic mixture", default= 2022)
parser.add_argument("--num_libs", help="number of synthtic libraries to create", default= 1)
parser.add_argument("--lib_size", help="number of synthtic libraries to create", default= 256000)
#parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = ['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','Weide', 'Ground', 'Shadow'])
parser.add_argument("--tree_index", help="labels of the tree species/classes in the correct order", default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--tree_class_weights", help="labels of the tree species/classes in the correct order", default = '[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1]')
parser.add_argument("--mixture_list", help="list of mixing complexity - how many classes can be mixed in one mixture", default = '[1,2,3]' )
parser.add_argument("--mixture_weights", help="wheight for every mixing complexity - For example [1, 1, 5, 1] will increase more chances to have 3-class mixtures", default = '[1, 5, 5]' )
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

def mixing(year,model_number):
    print('version ' + str(model_number))
    x_pure = make_array(os.path.join(args.working_directory, '1_pure' , f'samples_x{str(args.year)}'))
    x_pure = x_pure.astype(np.float32)
    print(x_pure.shape)
    y_pure = make_array(os.path.join(args.working_directory, '1_pure' , f'samples_y{str(args.year)}')),

    # check about nodata values / negative values
    mask = np.any(x_pure < 0, axis=(1, 2))
    print(y_pure[0][mask])
    x_pure = x_pure[~mask]
    y_pure = y_pure[0][~mask]

    indices_with_negatives = np.where(mask)[0]
    print("Positionen mit negativen Werten:", indices_with_negatives)

    training_sample = int(args.lib_size)
    x_mixed = []
    y_mixed = []
    index_list = list(range(len(y_pure)))
    lc_index = ast.literal_eval(args.tree_index)
    for _ in tqdm(range(training_sample)):
    #for _ in tqdm(range(10)):
        k = random.choices(ast.literal_eval(args.mixture_list), 
                           weights= ast.literal_eval(args.mixture_weights), k=1 )[0]
        fractions = np.random.dirichlet(np.ones(k),size=1)[0]
        # old: simple random selection of k samples
        # chosen_index = random.sample(index_list, k=k)

        # new: random selection of classes followed by selectino of samples within these classes
        # chose classes randomly
        #chosen_classes = random.sample(lc_index, k=k)
        chosen_classes = random.choices(lc_index, k=k, weights= ast.literal_eval(args.tree_class_weights))
        #print(chosen_classes,chosen_classes2)
        selected_indices = []
        # select index for each class
        for cls in chosen_classes:
            indices = np.where(y_pure == cls)[0]
            selected_idx = np.random.choice(indices)
            selected_indices.append(selected_idx)
        chosen_index = selected_indices
        
        x = 0
        y = np.zeros(len(lc_index))
        for i in range(len(chosen_index)):
            x += x_pure[chosen_index[i]]*fractions[i]
            label_pos = lc_index.index(y_pure[chosen_index[i]])
            y[label_pos] += fractions[i]
        x_mixed.append(x)
        y_mixed.append(y)
    x_mixed = np.array(x_mixed, np.float32)
    y_mixed = np.array(y_mixed, np.float32)
    count_greater_zero = (y_mixed > 0).sum(axis=0)
    print(count_greater_zero)

    if not os.path.exists(os.path.join(args.working_directory, '2_mixed_data' ,'version' +str(model_number))):
        os.makedirs(os.path.join(args.working_directory, '2_mixed_data','version' +str(model_number)))
    x_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data','version' +str(model_number), 'x_mixed_' + str(year) + '.npy')
    y_mixed_out_path = os.path.join(args.working_directory, '2_mixed_data','version' +str(model_number), 'y_mixed_' + str(year) + '.npy')
    print(x_mixed_out_path)
    #np.save(x_mixed_out_path, arr=x_mixed)
    #np.save(y_mixed_out_path, arr=y_mixed)

if __name__ == '__main__':
    for i in range(int(args.num_libs)):
        mixing(args.year,i+1)