#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 2024
@author: klehr
"""
# for paralellization 
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
import os

def set_threads(
    num_threads,
    set_blas_threads=True,
    set_numexpr_threads=True,
    set_openmp_threads=False
):
    num_threads = str(num_threads)
    if not num_threads.isdigit():
        raise ValueError("Number of threads must be an integer.")
    if set_blas_threads:
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["MKL_NUM_THREADS"] = num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    if set_numexpr_threads:
        os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    if set_openmp_threads:
        os.environ["OMP_NUM_THREADS"] = num_threads

set_threads(1)

# start pytohn code
from osgeo import gdal
import numpy as np
import os
import rasterio
from rasterio.mask import mask
from joblib import Parallel, delayed
from time import process_time
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2022")
parser.add_argument("--noisy_th", help="threshold for noisy prediction value", default= "20")
parser.add_argument("--use_shadow_th", help="should a special shadow threshold be used?", default= "F")
parser.add_argument("--shadow_th", help="threshold for shadow prediction value", default= "30")

parser.add_argument("--forest_mask_folder", help="path to the forest mask", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/cube")
parser.add_argument("--forest_mask_name", help="name of the forest mask raster", default= "holzbodenkarte_2018.tif")

parser.add_argument("--use_disturbance_mask", help="should a disturbance/bb maks be used?", default= "T")
parser.add_argument("--disturbance_mask_folder", help="path to the forest mask", default= '/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/hungry-beetle/HungryBeetle_DEU')
parser.add_argument("--disturbance_mask_name", help="name of the forest mask raster", default= 'disturbance_year.tif')

parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")
parser.add_argument("--num_models", help="number of models you want to create", default= 10)
parser.add_argument("--year", help="number of models you want to create", default= '2022')
parser.add_argument("--tile", help="The tile to be normalize", default= 'X0057_Y0057')
args = parser.parse_args()


def normalize_bands(tile):
    input_path = os.path.join(args.working_directory, '4_prediction', tile,'fraction_{year}.tif'.format(year = int(args.year)))
    #input_path = os.path.join(params['INPUT_FRACTION_DIR'], tile,'prediction_{year}_model1.tif'.format(year = params['YEAR']))
    output_path = os.path.join(args.working_directory, '5_prediction_normalized', tile)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the input raster
    gdal.DontUseExceptions()
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    if dataset is None:
        print("Failed to open the input raster.")
        return

    num_bands = dataset.RasterCount
    band_values = []
    for band_num in range(1, num_bands + 1):
        band = dataset.GetRasterBand(band_num)
        band_array = band.ReadAsArray()
        band_array[(band_array > 0) & (band_array <= int(args.noisy_th) )] = 0
        #print(band_num == (num_bands))       
        if ((args.use_shadow_th == 'T') & (band_num == num_bands)):
            band_array[(band_array > 0) & (band_array <= int(args.shadow_th) )] = 0
        band_array[band_array == 255] = 0
        #print(band_array.shape)
        band_values.append(band_array)

    # Calculate the sum of all band values for each pixel
    total_sum = sum(band_values)

    driver = gdal.GetDriverByName("GTiff") 
    new_dataset = driver.Create(os.path.join(output_path , '2_tree_fraction_norm_clip.tif' ),
                                dataset.RasterXSize, dataset.RasterYSize, num_bands, 
                                gdal.GDT_Byte, options=['COMPRESS=LZW'])

    # Copy geotransform and projection from the input dataset
    new_dataset.SetGeoTransform(dataset.GetGeoTransform())
    new_dataset.SetProjection(dataset.GetProjection())

    # load forest mask
    mask_path = os.path.join(args.forest_mask_folder, tile ,args.forest_mask_name ) 
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)
        # 0 = no forest; 1 = forest

    # load disturbance mask
    bb_mask_path = os.path.join(args.disturbance_mask_folder, tile, 'disturbance' ,args.disturbance_mask_name ) 
    with rasterio.open(bb_mask_path) as mask_src:
        bb_mask = mask_src.read(1)

    for band_num in range(num_bands):
        normalized_band = band_values[band_num] / total_sum
        #print(band_values[band_num][0,0], band_values[band_num][1,0], band_values[band_num][2,0] )
        
        
        rounded_band = np.round(normalized_band * 100).astype(np.uint8)
        # clip to forest mask
        rounded_band[mask==0] = 255
        rounded_band[mask==255] = 255
        #print(rounded_band[0,0], rounded_band[1,0], rounded_band[2,0] )
        #print('---')
        # write band to new raster
        new_band = new_dataset.GetRasterBand(band_num + 1)       
        new_band.WriteArray(rounded_band)
        new_band.SetNoDataValue(255)

    new_dataset.FlushCache()
    new_dataset = None
    dataset = None

    # step 2 BB mask 
    with rasterio.open(os.path.join(output_path , '2_tree_fraction_norm_clip.tif' )) as src:
        out_meta = src.meta
        normalized_raster = src.read()
        ground_exclude = np.copy(normalized_raster)
        band_ground = normalized_raster[-1]
        if (args.use_disturbance_mask == 'T'):
            #normalized_raster[:, bb_mask==1] = 255
            normalized_raster[:, (bb_mask > 1) & (bb_mask <= 2021)] = 255
        
    out_meta.update({"driver": "GTiff",
                 "compress":'lzw',
                 "dtype": rasterio.uint8
                 })


    # bandnames for normalized_raster
    with rasterio.open(os.path.join(output_path , '2_tree_fraction_norm_clip.tif'), "w" , **out_meta) as dest:
        dest.descriptions = tuple( ast.literal_eval(args.tree_labels) )
        dest.write(normalized_raster)

if __name__ == "__main__":
    #t1_start = process_time() 
    #tiles_to_normailze = []
    #for dir in os.listdir(os.path.join(args.working_directory, '4_prediction')):
    #    if dir.startswith('X0'):
    #        tiles_to_normailze.append(dir)

    if not os.path.exists(os.path.join(args.working_directory, '5_prediction_normalized') ):
        os.makedirs(os.path.join(args.working_directory, '5_prediction_normalized')  )

    #tiles_normalized =[]
    #for folder in os.listdir(os.path.join(args.working_directory, '5_prediction_normalized') ):
    #    if str(folder).startswith('X00') and os.path.isfile(os.path.join(args.working_directory, '5_prediction_normalized', folder, '2_tree_fraction_norm_clip.tif')):
    #        tiles_normalized.append(str(folder))

    #tiles = list(set(tiles_to_normailze) - set(tiles_normalized))
    #tiles = list(set(tiles_to_normailze))
    normalize_bands(args.tile)

    #Parallel(n_jobs=20)(delayed(normalize_bands)(tile) for tile in tiles)
    #print('Done')
