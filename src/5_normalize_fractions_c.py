#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 2024
@author: klehr
"""
import os
from joblib import Parallel, delayed

from osgeo import gdal
import numpy as np
import os
import rasterio
from rasterio.mask import mask
from time import process_time
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_c_noMapPap" )
parser.add_argument("--noisy_th", help="threshold for noisy prediction value", default= "20")

parser.add_argument("--forest_mask_folder", help="path to the forest mask", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/cube")
parser.add_argument("--forest_mask_name", help="name of the forest mask raster", default= "holzbodenkarte_2018.tif")

parser.add_argument("--use_disturbance_mask", help="should a disturbance/bb maks be used?", default= "T")
# parser.add_argument("--disturbance_mask_folder", help="path to the forest mask", 
#                    default= '/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/hungry-beetle/HungryBeetle_DEU')
# parser.add_argument("--disturbance_mask_name", help="name of the forest mask raster", default= 'disturbance_year.tif')
parser.add_argument("--disturbance_mask_folder", help="path to the forest mask", 
                    default= '/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/hungry-beetle/RefTh300_DistTh300_StdTh4')
parser.add_argument("--disturbance_mask_name", help="name of the forest mask raster", default= 'disturbances.tif')

parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", 
                    #default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    #default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Birke','Erle','OtherDT', 'Ground', 'Shadow']")
parser.add_argument("--year", help="number of models you want to create", default= '2021')
parser.add_argument("--tile", help="The tile to be normalize", default= 'X0055_Y0053')
parser.add_argument("--local", help="check if model is calculated for a local tileset", default= 'FALSE')
args = parser.parse_args()


def normalize_bands(tile):
    if args.local == 'TRUE':
        input_path = os.path.join(args.working_directory, '4_prediction', tile,'{year}_fraction.tif'.format(year = int(args.year)))
        output_path = os.path.join(args.working_directory, '5_prediction_normalized', tile)
    else:
        input_path = os.path.join(args.working_directory, '4_prediction_glob', tile,'{year}_fraction.tif'.format(year = int(args.year)))
        #output_path = os.path.join(args.working_directory, '5_prediction_normalized_glob', tile)
        output_path = os.path.join(args.working_directory, '5_prediction_normalized_newHB', tile)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.isfile(input_path):
        print(f'No prediction raster for tile {tile}, skipping normalization!')
        return    
    # ----------------- load forest mask -----------------
    mask_path = os.path.join(args.forest_mask_folder, tile ,args.forest_mask_name ) 
    if not os.path.isfile(mask_path):
        print(f'No forest mask for tile {tile}, skipping normalization!')
        return
    with rasterio.open(mask_path) as mask_src:
        forest_mask = mask_src.read(1)
        # 0 = no forest; 1 = forest
    
    # ----------------- load disturbance mask -----------------
    # bb_mask_path = os.path.join(args.disturbance_mask_folder, tile, 'disturbance' ,args.disturbance_mask_name ) 
    # if not os.path.isfile(bb_mask_path):
    #     print(f'No disturbance mask for tile {tile}, skipping normalization!')
    #     return
    # with rasterio.open(bb_mask_path) as mask_src:
    #     bb_mask = mask_src.read(1)
    bb_mask_path = os.path.join(args.disturbance_mask_folder, tile ,args.disturbance_mask_name )
    if not os.path.isfile(bb_mask_path):
        print(f'No disturbance mask for tile {tile}, skipping normalization!')
        os.rmdir(output_path)
        return
    with rasterio.open(bb_mask_path) as mask_src:
        bb_mask = mask_src.read(2)
        
    # ----------------- load data and calculate sum -----------------
    with rasterio.open(input_path) as src:
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                 "compress":'ZSTD',
                 "dtype": rasterio.uint8
                 })
        band_values = []
        for band_num in range(1, src.count + 1):
            band_array = src.read(band_num)
            band_array[(band_array > 0) & (band_array <= int(args.noisy_th) )] = 0
            band_array[band_array == 255] = 0
            band_values.append(band_array)

        # Calculate the sum of all band values for each pixel
        band_stack = np.stack(band_values)
        total_sum = np.sum(band_stack, axis=0)

    # ----------------- normalize each band and store it in the output -----------------
        with rasterio.open(os.path.join(output_path , '2_tree_fraction_norm_clip.tif'), "w" , **out_meta) as dest:
            dest.descriptions = tuple( ast.literal_eval(args.tree_labels) )
            for band_num in range(src.count):
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized_band = band_stack[band_num] / total_sum
                normalized_band[total_sum == 0] = 255
                #normalized_band = band_values[band_num] / total_sum
                rounded_band = np.round(normalized_band * 100)
                # clip to forest mask
                rounded_band[forest_mask==0] = 255
                rounded_band[forest_mask==255] = 255 # outside germany
                rounded_band = rounded_band.astype(np.uint8)
                # clip to disturbance mask
                if (args.use_disturbance_mask == 'T'):
                    rounded_band[(bb_mask > 1) & (bb_mask <= 2021)] = 255
                dest.write(rounded_band, band_num+1)

if __name__ == "__main__":
    #normalize_bands(args.tile)
    # ------ global model -------
    #tiles_to_normalize = []
    #for dir in os.listdir(os.path.join(args.working_directory, '4_prediction_glob')):
    #    if dir.startswith('X0'):
    #        tiles_to_normalize.append(dir)
    #tiles = list(set(tiles_to_normalize))
    #Parallel(n_jobs=20)(delayed(normalize_bands)(tile) for tile in tiles)
    #----------------------------
    #------- Only One Federal state (RLP) ------
    with open('/data/ahsoka/eocp/forestpulse/02_scripts/spline/fed_states_tiles/BW_tiles_all.txt', 'r') as file:
       # Read all lines into a list
       lines = file.readlines()
    tiles = [line.strip() for line in lines]
    tiles = tiles[1:] # remove header
    print(tiles)
    Parallel(n_jobs=20)(delayed(normalize_bands)(tile) for tile in tiles)
    