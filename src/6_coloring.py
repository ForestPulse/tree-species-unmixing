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

from joblib import Parallel, delayed
# start pytohn code
import rasterio
import numpy as np
import colorsys
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime_2nd_sampling2")
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Ground', 'Shadow']")

args = parser.parse_args()

def hsv_to_rgb(h, s, v):
    # Umwandlung auf den [0,1] Bereich für Hue
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return r, g, b

# Pfad zur GeoTIFF-Datei


bands = ast.literal_eval(args.tree_labels)

def color_raster(working_dir, tile, no_of_tile, length_list):

    print('-------- Start coloring ' + tile + ' | ' + str(no_of_tile) + f'/{str(length_list)} ----------')
    if not tile.startswith('X'):
       print('Not tile, skipping!')
       return

    tiff_path = os.path.join(working_dir, tile , '2_tree_fraction_norm_clip.tif')
    tiff_path
    # Definiere das Ausgabe-GeoTIFF
    output_tiff_path = os.path.join(working_dir, tile , '2_tree_fraction_colored.tif')

    if not os.path.isfile(os.path.join(working_dir, tile,'2_tree_fraction_norm_clip.tif')):
        print('folder ' + tile + 'is empty.')
        return

    with rasterio.open(tiff_path) as src:

        # Meta-Daten
        crs = src.crs
        transform = src.transform
        profile = src.profile

        # Als numpy array laden (alle Bänder)
        descriptions = src.descriptions

        #nodatamask
        mask = src.read(1)
          
        # OtherDT
        #OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Birch', 'Willow', 'Robinia', 'Poplar']]
        #OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Birke', 'Weide', 'Pappel']]
        OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Pappel', 'OtherDT']]
        selected_bands = [src.read(i) for i in OtherDT_band_indexes]
        #print(np.stack(selected_bands, axis =0).shape)
        Other_DT = np.sum(selected_bands, axis=0)  # shape: (height, width)
        #print(Other_DT.shape)
        Other_DT_expanded = Other_DT[np.newaxis, :, :]

        # Background
        background_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Ground', 'Shadow']]
        selected_bands = [src.read(i) for i in background_band_indexes]
        #print(np.stack(selected_bands, axis =0).shape)
        background = np.sum(selected_bands, axis=0)  # shape: (height, width)

        #Array
        #band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in  ['Beech', 'Oak', 'Alder', 'Maple', 'Pine', 'Spruce', 'Douglas', 'Larch']]
        band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in  ['Buche', 'Eiche','Birke','Erle', 'Ahorn', 'Kiefer', 'Fichte', 'Douglasie', 'Larche', 'Tanne']]
        selected_bands = [src.read(i) for i in band_indexes]
        data = np.stack(selected_bands, axis =0)
        #print(data.shape)

        # Anhängen entlang der Band-Achse (axis=0)
        data = np.concatenate((data, Other_DT_expanded), axis=0)
        #print(data.shape)
        # Maximalwert je Pixel (über axis=0, also über die Bänder)
        max_values = np.max(data, axis=0)  # shape: (3000, 3000)
        # Index des Bandes mit dem Maximalwert je Pixel
        max_indices = np.argmax(data, axis=0)

        #---------------------------
        # HSV array
        value = 1 - (background / 100)
        saturation = max_values / 100
        #hue_map = {0: 120, 1: 60, 2: 190, 3: 25, 
        #            4: 0, 5: 240, 6: 320, 7: 45, 8: 160}
        hue_map = {0: 240,   1:0  , 2:280 ,   3:320   ,    4: 45  , 5:120 , 6:60  ,    7:25   ,  8: 210  ,   9:180  , 10: 160}  
                #  Fichte,  Kiefer| Tanne | Douglasie |  Lärche   | Buche | Eiche | Ahorn     |   Birke  | Erle     | otherDT
                #   blau |    rot | lila  |    pink   | h. orange |  grün | gelb  | d. orange | m.  blau | hellblau | türkis
        hue = np.array([[hue_map[idx] for idx in row] for row in max_indices])

        hsv_array = np.stack([hue, saturation, value], axis=0)

        rgb_array = np.zeros_like(hsv_array)
        #for i in tqdm(range(hsv_array.shape[1])):
        for i in range(hsv_array.shape[1]):
            for j in range(hsv_array.shape[2]):
                h, s, v = hsv_array[:, i, j]  # Hole Hue, Saturation und Value
                r, g, b = hsv_to_rgb(h, s, v)
               
                rgb_array[0, i, j] = int(r*254)  # Red-Kanal
                rgb_array[1, i, j] = int(g*254)  # Green-Kanal
                rgb_array[2, i, j] = int(b*254)  # Blue-Kanal
        # no data assignment
        rgb_array[0][mask==255]=255
        rgb_array[1][mask==255]=255
        rgb_array[2][mask==255]=255

        with rasterio.open(output_tiff_path, 'w',  driver='GTiff', nodata=255,
                    height=rgb_array.shape[1], width=rgb_array.shape[2], 
                    count=3, dtype='uint8', crs=crs, transform=transform, compress="zstd") as dst:
            dst.write(rgb_array[0], 1)  # Rot-Kanal in Band 1
            dst.write(rgb_array[1], 2)  # Grün-Kanal in Band 2
            dst.write(rgb_array[2], 3)  # Blau-Kanal in Band 3

    print('-------- Coloring ' + tile + ' done successfully  | ' + str(no_of_tile) + f'/{str(length_list)} ----------')


if __name__ == '__main__':
    working_dir = os.path.join(args.working_directory, '5_prediction_normalized')
    list_tile = os.listdir(os.path.join(args.working_directory, '5_prediction_normalized') )

    tiles_to_color =[]
    for folder in os.listdir(os.path.join(args.working_directory, '5_prediction_normalized') ):
        if (str(folder).startswith('X00')) and not (os.path.isfile(os.path.join(args.working_directory, '5_prediction_normalized',folder,'2_tree_fraction_colored.tif'))) and (os.path.isfile(os.path.join(args.working_directory, '5_prediction_normalized',folder,'2_tree_fraction_norm_clip.tif'))):
            tiles_to_color.append(str(folder))
    list_tile = tiles_to_color

    Parallel(n_jobs=25)(delayed(color_raster)(working_dir, tile, list_tile.index(tile),len(list_tile)) for tile in list_tile)