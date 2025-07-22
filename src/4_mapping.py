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
import tensorflow as tf
import numpy as np
from osgeo import gdal
import rasterio
import ast
import argparse
from datetime import datetime
from keras.saving import register_keras_serializable
from joblib import Parallel, parallel_backend, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermalTime_Spline" )
parser.add_argument("--working_directory", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime_2nd_sampling")
parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")
parser.add_argument("--num_models", help="number of models you want to create", default= '5')
parser.add_argument("--year", help="number of models you want to create", default= '2021')
args = parser.parse_args()

@register_keras_serializable(package="Custom")
class SumToOneLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    
def get_stack(tile, year):
    def get_band(band_name):# get all time-steps for one sentinel-2 band
        # get all paths for every date
        band_paths =[]
        for datei in os.listdir(os.path.join(args.dc_folder,tile)):
            if ('_'+band_name+'_' in datei) and datei.endswith('.tif'):
                path = os.path.join(args.dc_folder,tile, datei)
                band_paths.append(path)   
        band_paths = sorted(band_paths)
        # load all dates for the called band
        list_of_dates = []
        for band_date in band_paths:
            with rasterio.open(band_date) as src:
                date = src.read()
            date = np.moveaxis(date, 0, -1)
            list_of_dates.append(date) 
        # and convert it to the right numpy array shape
        list_of_dates = np.array(list_of_dates)
        list_of_dates = np.squeeze(list_of_dates, axis=-1)
        list_of_dates = np.moveaxis(list_of_dates, 0, -1)
        return list_of_dates
    
    band_list = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNIR', 'NIR', 'SW1', 'SW2']
    stack = np.array([get_band(b) for b in band_list])
    stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, no_of_tile, length):
    @register_keras_serializable(package="Custom")
    class SumToOneLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
        
    def pred(model, x):
        y_pred = model(x, training=False)
        return y_pred.numpy()
    def norm(a):
        a_out = a/10000.
        return a_out
    def toRasterFraction(arr_in, name_list):
        y1 = int(args.year)-2
        y2 = int(args.year)
        path = os.path.join(args.dc_folder, tile, '{y1}-{y2}_001-365_HL_UDF_SEN2L_RSP_BLU_010.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(args.working_directory, '4_prediction', tile, 'fraction_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Byte)
        #outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(255)
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterDeviation(arr_in, name_list):
        y1 = int(args.year)-2
        y2 = int(args.year)
        path = os.path.join(args.dc_folder, tile, '{y1}-{y2}_001-365_HL_UDF_SEN2L_RSP_BLU_010.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(args.working_directory, '4_prediction', tile, 'deviation_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(255)
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterClassification(arr_in):
        y1 = int(args.year)-2
        y2 = int(args.year)
        path = os.path.join(args.dc_folder, tile, '{y1}-{y2}_001-365_HL_UDF_SEN2L_RSP_BLU_010.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        path_out = os.path.join(args.working_directory, '4_prediction', tile, 'classification_' + year + '.tif')
        outdata = driver.Create(path_out, rows, cols, 1, gdal.GDT_Byte)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr_in)
        outdata.GetRasterBand(1).SetNoDataValue(255)
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def predict_model_on_tile(model, x_in, nodata_mask):
        H, W, T, B = x_in.shape
        x_flat = x_in.reshape((-1, T, B))  # Shape: (H×W, 28, 10)

        # Optional: Nur gültige Pixel auswählen (spart Zeit)
        valid_mask = ~nodata_mask.flatten()
        x_valid = x_flat[valid_mask]

        # Prüfen, ob gültige Pixel vorhanden sind
        if x_valid.shape[0] == 0:
            # Keine gültigen Daten, also leere Vorhersage zurückgeben
            return np.zeros((H, W, model.output_shape[-1]), dtype=np.float32)

        # Mit tf.data schneller
        ds = tf.data.Dataset.from_tensor_slices(x_valid).batch(1024)
        preds = model.predict(ds, verbose=0)  # Shape: (valid_samples, N_CLASSES)

        # Rückkonvertieren ins Bildformat
        y_pred = np.zeros((H * W, preds.shape[-1]), dtype=np.float32)
        y_pred[valid_mask] = preds
        y_pred = y_pred.reshape((H, W, -1))  # Shape: (H, W, N_CLASSES)
        return y_pred
    
    # =============================================
    # load model list
    # =============================================
    model_list = []
    for i in range(int(args.num_models)):
    #for i in [1]:
        model_path = os.path.join(args.working_directory, '3_trained_model', 'version' +str(i+1), 'saved_model'+ str(i+1)+ '.keras') 
        model = tf.keras.models.load_model(model_path)
        model_list.append(model)

    # =============================================
    # define input (if present) and output
    # =============================================
    y1 = int(args.year)-2
    y2 = int(args.year)
    blue_band = os.path.join(args.dc_folder, tile, '{y1}-{y2}_001-365_HL_UDF_SEN2L_RSP_BLU_010.tif'.format(y1=y1, y2=y2) )
    if not os.path.isfile(blue_band):
        print('Not tile, skipping!')
        return
    out_dir = os.path.join(args.working_directory, '4_prediction', tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # start processing here
    print('Predicting', tile, year, '...', str(no_of_tile+1) , '/{number} ----------'.format(number=length),sep = ' ')
    start=datetime.now()

    x_in = get_stack(tile, year)
    nodata_mask = x_in[:, :, 0, 0] == -9999
    x_in = norm(x_in.astype(np.float32))
   
    y_out = np.zeros([x_in.shape[0], x_in.shape[1], len(ast.literal_eval(args.tree_class_list))]) 
    name_list = ast.literal_eval(args.tree_labels)

    # =============================================
    #          multi model prediction
    # =============================================
    list_predictions =[]
    model_num = 0 
    for model in model_list:
        y_out = predict_model_on_tile(model, x_in, nodata_mask)
        y_out = np.clip(y_out * 100, 0, 100)
        list_predictions.append(np.copy(y_out))
        #model_num =  model_num + 1
        #for i in range(y_out.shape[1]):
        #    x_batch = x_in[i, ...]
        #    y_out[i, ...] = pred(model, x_batch)
        #y_out = y_out * 100
        #y_out[y_out < 0.] = 0.
        #y_out[y_out > 100.] = 100.
        #list_predictions.append(np.copy(y_out))
        #single = y_out
        #single[nodata_mask] = 255
    stacked_arrays = np.stack(list_predictions, axis=-1)
    # median fraction and deviation
    #average_array = np.mean(stacked_arrays, axis=-1)
    median_array = np.median(stacked_arrays, axis=-1)
    #deviation = np.mean(np.absolute(stacked_arrays - average_array[..., np.newaxis]), axis=-1)

    # classification of dominant species
    y_out_clf = np.argmax(median_array, axis= -1)
    y_out_clf += 1
    y_out_clf[nodata_mask] = 255
    y_out_clf = y_out_clf.astype(np.int8)

    # ===============
    # writing outputs 
    # ===============
    # median
    median_array[nodata_mask] = 255
    median_array = median_array.astype(np.int8)   
    toRasterFraction(median_array, name_list)
    # deviation
    #deviation[nodata_mask] = 255
    #deviation = deviation.astype(np.int8)
    #toRasterDeviation(deviation, name_list)
    
    toRasterClassification(y_out_clf)
    print('-------- Predicting ' + tile + ' done successfully  | ' + str(no_of_tile+1) + '/{number} | Duration: '.format(number=length) + str(datetime.now()-start) + ' ----------')

if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.working_directory, '4_prediction')):
        os.makedirs(os.path.join(args.working_directory, '4_prediction'))

    list_tiles = []
    for folder in os.listdir(args.dc_folder):
        if str(folder).startswith('X00'):
            list_tiles.append(str(folder))
    list_predicted =[]
    for folder in os.listdir(os.path.join(args.working_directory, '4_prediction')):
        if str(folder).startswith('X00'):
            list_predicted.append(str(folder))

    list_tiles = list(set(list_tiles) - set(list_predicted))
    year = int(args.year)
    #for tile in list_tiles:
    #    predict(tile, '2021', model_list, list_tiles.index(tile), len(list_tiles))
    Parallel(n_jobs=10, backend="loky")(delayed(predict)(
        tile, '2021', list_tiles.index(tile), len(list_tiles)) for tile in list_tiles)