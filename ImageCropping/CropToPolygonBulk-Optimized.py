# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CropToPolygonBulk.py --inputfile_path /export/archive/input.csv

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import json
import csv
import pandas as pd
import CropPolygons.CropPolygonsToSingleImage as CropPolygonsToSingleImage
import CropPolygonsSquareRectangles.CropPolygonsToSingleSquareRectangularImage as CropPolygonsToSingleSquareRectangularImage

import time
import os

DEBUG = os.environ.get('DEBUG')
if DEBUG=="1" or str(DEBUG).lower() == "true":
    DEBUG = True
    print(f"DEBUG enabled {DEBUG}")
else:
    DEBUG = False

start_time = time.time()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputfile_path", required=True, help="complete file path to the image you want to crop to a polygon")
args = vars(ap.parse_args())

#/home/production/cv/bin/python3 /home/production/cxgn/DroneImageScripts/ImageCropping/CropToPolygonBulk.py 
#--inputfile_path 
# \'/home/production/cxgn/sgn//static/documents/tempfiles/drone_imagery_plot_polygons/bulkinputbAJ1\'';
inputfile_path = args["inputfile_path"]
#inputfile_path = '~/data/imagebreed_test/polygon_input/bulkinput0vIJ.tsv'

# +
#input_image_file_data = pd.read_csv(inputfile_path, sep="\t", header=None)

# +
#input_image_file_data.columns = ['inputfile_path', 'outputfile_path', 
#                                  'polygon_json', 'polygon_type', 'image_band_index']

input_image_file_data = pd.read_csv(inputfile_path, sep="\t", 
            names= ['inputfile_path', 'outputfile_path', 'polygon_json', 
                    'polygon_type', 'image_band_index'])


# +
def separate_image_band(img, image_band_index):
    img_shape = img.shape
    print("Separate band -> dimensions: ", len(img_shape))
    print("Band index", image_band_index)
    
    if len(img_shape) == 3:
        if img_shape[2] == 3:
            b,g,r = cv2.split(img)
            if image_band_index is not None and not np.isnan(image_band_index):
                image_band_index = int(image_band_index)
                if image_band_index == 0:
                    img = b
                if image_band_index == 1:
                    img = g
                if image_band_index == 2:
                    img = r
    return img


def get_cropped_image(img, polygon_type, polygons):
    if polygon_type == 'rectangular_square':
        sd = CropPolygonsToSingleSquareRectangularImage.CropPolygonsToSingleSquareRectangularImage()
        finalImage = sd.crop(img, polygons)
    elif polygon_type == 'rectangular_polygon':
        sd = CropPolygonsToSingleImage.CropPolygonsToSingleImage()
        finalImage = sd.crop(img, polygons)
    return finalImage


def load_image(inputfile_path, image_band_index):
    img = cv2.imread(inputfile_path, cv2.IMREAD_UNCHANGED)
    img_shape = img.shape
    if len(img_shape) == 3:
        img = separate_image_band(img, image_band_index)
    return img



#def process_dataframe(input_image_file_data):

def process_row(row, img):
    #for index, row in input_image_file_data.iterrows():
    #print(row)
    inputfile_path = row[0]
    outputfile_path = row[1]
    polygon_json = row[2]
    polygon_type = row[3]
    image_band_index = row[4] #row.iloc[4] 
    polygons = json.loads(polygon_json)

    # this is to retain compatibility with the original script
    # we would rather read the input image once if all of them are the same
    if img is None:
        print("loading image", inputfile_path)
        img = load_image(inputfile_path, image_band_index)
  
    finalImage = get_cropped_image(img, polygon_type, polygons)
    if DEBUG:
       print(f"Saving slice {outputfile_path}")
    
    cv2.imwrite(outputfile_path, finalImage)
    #cv2_imshow("Res", finalImage)
    return outputfile_path

# this asserts that the index is in the record
# def process_record(record, img):
#     index = record[0]
#     row = record[1]
#     #import time
#     #time.sleep(1)
#     return process_row(row, img=img)

# for some images is faster without the Multi-processing
USE_MILTIPROCESSING = False
N_THREADS = 5
#SHARED_IMAGES = False 
#image_cache = {}

#if the inputs are the same we read the source image once

#check how many input images are loaded
unique_inputs = input_image_file_data.inputfile_path.unique()
unique_bandindexes = input_image_file_data.image_band_index.unique()
if len(unique_inputs) == 1 and len(unique_bandindexes)==1:
   # SHARED_IMAGES = True
    inputfile_path = unique_inputs[0]
    image_band_index = unique_bandindexes[0]
    print(f"Single Input Image: {inputfile_path}\n will try to uses shared image.")
    IMAGE = load_image(inputfile_path, image_band_index)

else:
    IMAGE = None
    print("Seems like there are multiple input images.")


if USE_MILTIPROCESSING:
    import multiprocessing
    def task(record_img_tuple):
        #process_record(record, img=img)
        record, img = record_img_tuple
        #return process_record(record, img)
        process_row(record, img)
        
    with multiprocessing.Pool(N_THREADS) as mp:
        records = list(input_image_file_data.itertuples(index=False, name=None))
        #out = mp.map(task, [(record, IMAGE) for record in input_image_file_data.iterrows()])
        out = mp.map(task, [(record, IMAGE) for record in records])

else:
    out = []
    # for record in list(input_image_file_data.iterrows()):
    #     out.append(process_record(record, IMAGE))
    for record in list(input_image_file_data.itertuples(index=False)):
        #out.append(process_record(record, IMAGE))
        out.append(process_row(record, IMAGE))
    #def task(record, img=None):
    #    process_record(record, img=img)
    #out = list(map(task, list(input_image_file_data.iterrows())))
total_processed = len(set(out))
total_seconds = time.time() - start_time
print(f"{total_processed=} in {total_seconds:.4f} seconds.")

#print(list(out))

# +


# from matplotlib import pyplot as plt

# def cv2_imshow(title, image, axes=None, **kwargs):
#     a = image.clip(0, 255).astype('uint8')
#     plt.axis('off')

#     # cv2 stores colors as BGR; convert to RGB
#     if a.ndim == 3:
#         if a.shape[2] == 4:
#             a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
#         else:
#             a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#     if axes is not None:
#         axes.axis('off')
#         if title is not None:
#             axes.set_title(title)
#         #return axes.imshow(a, **kwargs)
#         im = axes.imshow(a, **kwargs)
#     else:
#         #plt.axis('off')
#         if title is not None:
#             plt.title(title)
#         im = plt.imshow(a, **kwargs)

#     #if save_path is not None:
#     #    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     # return im
#     #return plt.imshow(a, **kwargs)
#     return im 

