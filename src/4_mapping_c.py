#!/usr/bin/env python
import ast
import argparse
from datetime import datetime
#import sys
#from keras.saving import register_keras_serializable
#import shutil
import rasterio
import numpy as np
import os
import tensorflow as tf
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube",
                     default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC" )
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                     default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_c_noMapPap")
parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", 
                    #default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    # default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Birke','Erle','OtherDT', 'Ground', 'Shadow']")
parser.add_argument("--num_models", help="number of models you want to create", default= '10')
parser.add_argument("--year", help="number of models you want to create", default= '2021')
parser.add_argument("--tile", help="The tile to be predicted", default= 'X0055_Y0053')
parser.add_argument("--forest_mask", help="path to the forest mask raster", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/cube/") # holzbodenkarte_2018
                    #default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_CodeDE/")
parser.add_argument("--local", help="check if model is calculated for a local tileset", default= 'FALSE')
parser.add_argument("--tile_set", help="tile-set of the local model application", 
                    default= 'Local')
args = parser.parse_args()
  
def get_stack(tile, year):
    file_path = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')
    with rasterio.Env(GDAL_NUM_THREADS="1"):
        with rasterio.open(file_path) as src:
            stack = src.read()
            stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, no_of_tile, length, tile_set):
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def predict_model_on_tile(model, x_flat, forest_mask_flat, H, W):
        # Prüfen, ob gültige Pixel vorhanden sind
        if x_flat.shape[0] == 0:
            # Keine gültigen Daten, also leere Vorhersage zurückgeben
            return np.zeros((H, W, model.output_shape[-1]), dtype=np.float32)
        # Prediction
        ds = tf.data.Dataset.from_tensor_slices(x_flat).batch(1024)
        preds = model.predict(ds, verbose=0)  # Shape: (valid_samples, N_CLASSES)
        y_pred = np.zeros((H * W, preds.shape[-1]), dtype=np.float32)
        y_pred[forest_mask_flat == 1] = preds
        # Rückkonvertieren ins Bildformat
        y_pred = y_pred.reshape((H, W, -1))  # Shape: (H, W, N_CLASSES)
        return y_pred
    
    # =============================================
    # define input (if present) and output
    # =============================================
    input_raster = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')

    if not os.path.isfile(input_raster):
        print(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' Not tile - skipped!')
        print('Not tile, skipping!')
        return
    
    # start processing here
    print('Status ' + tile_set + ': ' + tile + f' [{no_of_tile}/{length}]' + ' prediction started...')
    start=datetime.now()
    x_in = get_stack(tile, year)

    # forest mask
    #mask_path = os.path.join(args.forest_mask, tile, f'RLP_forest_union.tif')
    mask_path = os.path.join(args.forest_mask, tile, f'holzbodenkarte_2018.tif')
    if not os.path.isfile(mask_path):
        print(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' No forest mask - skipped!')
        print('No forest mask, skipping!')
        return
    with rasterio.open(mask_path) as src:
        meta = src.meta.copy()
        forest_mask = src.read(1)
    forest_mask_flat = forest_mask.flatten()

    meta.update(
        count=len(ast.literal_eval(args.tree_labels)),  # Anzahl der Layer entspricht der Anzahl der TIFFs
        dtype='uint8',          # 8-Bit Integer
        compress='ZSTD'          # LZW-Komprimierung
    )

    # apply forest_mask
    H, W, bands = x_in.shape
    x_flat = x_in.reshape((-1, bands))
    x_valid = x_flat[forest_mask_flat == 1]
    #print('Reading Done - ',datetime.now()-start)

    # =============================================
    # load model list
    # =============================================
    from keras.saving import register_keras_serializable
    @register_keras_serializable(package="Custom")
    class SumToOneLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)

    model_list = []
    for i in range(int(args.num_models)):
    #for i in [1,2]:
        if args.local == 'TRUE':
            model_path = os.path.join(args.working_directory, '3_trained_model_tile', tile , 'version' +str(i+1), 'saved_model'+ str(i+1)+ '.keras')
        else:
            model_path = os.path.join(args.working_directory, '3_trained_model_glob', 'version' +str(i+1), 'saved_model'+ str(i+1)+ '.keras')
        model = tf.keras.models.load_model(model_path)
        model_list.append(model)
    #print('Model loading Done - ',datetime.now()-start)
  
    x_valid = norm(x_valid.astype(np.float32))

    y_out = np.zeros([H, W, len(ast.literal_eval(args.tree_class_list))]) 
    name_list = ast.literal_eval(args.tree_labels)
    #print('Data normed',datetime.now()-start) # at about 3:00

    # =============================================
    #          multi model prediction
    # =============================================
    list_predictions =[]
    model_num = 0 
    for model in model_list:
        y_out = predict_model_on_tile(model, x_valid, forest_mask_flat, H, W)
        #y_out = np.clip(y_out * 100, 0, 100)
        y_out = np.clip(y_out * 100, a_min = 0, a_max = None)
        list_predictions.append(np.copy(y_out))
    stacked_arrays = np.stack(list_predictions, axis=-1)

    #print('Prediction Done - ',datetime.now()-start) # at about 3:30 for 2 models
    # =============================================
    #          median fraction and deviation
    # =============================================
    average_array = np.mean(stacked_arrays, axis=-1)
    median_array = np.median(stacked_arrays, axis=-1)
    deviation = np.mean(np.absolute(stacked_arrays - average_array[..., np.newaxis]), axis=-1)
    # classification of dominant species
    y_out_clf = np.argmax(median_array, axis= -1)
    y_out_clf += 1
    y_out_clf[forest_mask != 1] = 255
    y_out_clf = y_out_clf.astype(np.int8)

    median_array[forest_mask != 1] = 255
    median_array = median_array.astype(np.int8)

    # =============================================
    #              writing outputs 
    # =============================================       
    if args.local == 'TRUE':
        out_dir = os.path.join(args.working_directory, '4_prediction', tile)
    else:
        out_dir = os.path.join(args.working_directory, '4_prediction_glob', tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. median fraction
    with rasterio.open(os.path.join(out_dir, f'{args.year}_fraction.tif'), 
                       'w', **meta) as dst:
            dst.descriptions = name_list
            for i in range(median_array.shape[-1]):
                dst.write(median_array[..., i].astype(np.uint8), i+1)
                dst.set_band_description(i+1, name_list[i])

    print('Status ' + tile_set + ': ' + tile + f' [{no_of_tile}/{length}]' + ' prediction finished...')
    

    #write_file(median_array, name = 'fraction', data_type = gdal.GDT_Byte, num_bands = median_array.shape[-1])
    # # --- classification ---
    # #toRasterClassification(y_out_clf)
    # write_file(y_out_clf, name = 'classification', data_type = gdal.GDT_Byte, num_bands = 1)

    # # ------ deviation ------
    # deviation[nodata_mask] = 255
    # deviation = deviation.astype(np.int8)
    # #toRasterDeviation(deviation, name_list)
    # write_file(deviation, name = 'deviation', data_type = gdal.GDT_Byte, num_bands = median_array.shape[-1])
    # #print('-------- Predicting ' + tile + '   [Done]  | ' + str(no_of_tile+1) + '/{number} | Duration: '.format(number=length) + str(datetime.now()-start) + ' ----------')
    # #print_at_line(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' prediction finished ... | Duration: ' + str(datetime.now()-start))
    # print( 'Status ' + tile_set + ': ' + tile + f' [{no_of_tile}/{length}]' + ' prediction finished ... | Duration: ' + str(datetime.now()-start))

if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.working_directory, '4_prediction_glob')):
        os.makedirs(os.path.join(args.working_directory, '4_prediction_glob'))
    if not os.path.exists(os.path.join(args.working_directory, '4_prediction')):
        os.makedirs(os.path.join(args.working_directory, '4_prediction'))
    # ---- Predict just one tile for testing ----
    #predict(args.tile, args.year , 1, 1, args.tile_set)
    #--------------------------------------------

    #------- Repredict all tiles ------
    # list_tiles = []
    # for folder in os.listdir(args.dc_folder):
    #     if str(folder).startswith('X00'):
    #         list_tiles.append(str(folder))
    # Parallel(n_jobs=10, backend="loky")(delayed(predict)(
    #     tile, args.year , list_tiles.index(tile)+1, len(list_tiles), args.tile_set) for tile in list_tiles)
    #------------------------------------

    #----------- Only Repredict tiles that are not present yet ------
    list_tiles = []
    for folder in os.listdir(args.dc_folder):
    #for folder in os.listdir(os.path.join(args.working_directory, '3_trained_model_tile')):
        if str(folder).startswith('X00') & (
            not os.path.isfile(os.path.join(args.working_directory, '4_prediction_glob' ,folder, f'{args.year}_fraction.tif'))):
            list_tiles.append(str(folder))
    Parallel(n_jobs=10, backend="loky")(delayed(predict)(
        tile, args.year , list_tiles.index(tile)+1, len(list_tiles), args.tile_set) for tile in list_tiles)
    #------------------------------------

    #------- Only One Federal state (RLP) ------
    # with open('/data/ahsoka/eocp/forestpulse/02_scripts/spline/fed_states_tiles/RLP_tiles_allY.txt', 'r') as file:
    #    # Read all lines into a list
    #    lines = file.readlines()
    # list_tiles = [line.strip() for line in lines]
    # list_tiles = list_tiles[1:] # remove header
    # Parallel(n_jobs=10, backend="loky")(delayed(predict)(
    #    tile, args.year , list_tiles.index(tile)+1, len(list_tiles), args.tile_set) for tile in list_tiles)
    #------------------------------------
    
