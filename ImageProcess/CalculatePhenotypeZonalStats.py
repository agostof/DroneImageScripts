# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/ImageProcess/CalculatePhenotypeZonalStats.py --image_paths /folder/mypic1.png,/folder/mypic2.png --results_outfile_path /folder/myresults.csv --image_band_index 0

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import statistics
from collections import defaultdict
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_paths", required=True, help="image paths comma separated")
ap.add_argument("-r", "--results_outfile_path", required=True, help="file path where results will be saved")
ap.add_argument("-j", "--image_band_index", required=True, help="file path where results will be saved")
args = vars(ap.parse_args())

input_images = args["image_paths"]
results_outfile = args["results_outfile_path"]
image_band_index = int(args["image_band_index"])
images = input_images.split(",")

result_file_lines = [
    ['nonzero_pixel_count', 'total_pixel_sum', 'mean_pixel_value', 'harmonic_mean_value', 'median_pixel_value', 'variance_pixel_value', 'stdev_pixel_value', 'pstdev_pixel_value', 'min_pixel_value', 'max_pixel_value', 'minority_pixel_value', 'minority_pixel_count', 'majority_pixel_value', 'majority_pixel_count', 'pixel_variety_count']
]

count = 0
for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img_shape = img.shape
    
    if len(img_shape) == 3:
        if img_shape[2] == 3:
            b,g,r = cv2.split(img)
            if image_band_index == 0:
                img = b
            if image_band_index == 1:
                img = g
            if image_band_index == 2:
                img = r

    non_zero = cv2.countNonZero(img)
    #print("Nonzero: %s" % non_zero)

    height, width = img.shape

    total_pixel_sum = 0
    pixel_array = []
    pixel_dict = defaultdict(int)

    for i in range(0, height):
        for j in range(0, width):
            px = int(img[i,j])
            total_pixel_sum += px
            pixel_array.append(px)
            pixel_dict[px] += 1

    #print("Total: %s" % total_pixel_sum)

    mean_pixel_value = statistics.mean(pixel_array)
    #print("Mean: %s" % mean_pixel_value)

    harmonic_mean_pixel_value = statistics.harmonic_mean(pixel_array)
    #print("Harmonic Mean: %s" % harmonic_mean_pixel_value)

    pixel_array_np = np.array(pixel_array)
    pixel_array_sort = np.sort(pixel_array_np)
    pixel_median_value = statistics.median(pixel_array_sort)
    #print("Median: %s" % pixel_median_value)

    pixel_variance = statistics.variance(pixel_array)
    #print("Variance: %s" % pixel_variance)

    pixel_standard_dev = statistics.stdev(pixel_array)
    #print("Stdev: %s" % pixel_standard_dev)

    pixel_pstandard_dev = statistics.pstdev(pixel_array)
    #print("Pstdev %s" % pixel_pstandard_dev)

    min_pixel = pixel_array_sort[0]
    max_pixel = pixel_array_sort[-1]
    #print("Min: %s" % min_pixel)
    #print("Max: %s" % max_pixel)

    pixel_sorted_by_value = sorted(pixel_dict.items(), key=lambda kv: kv[1])
    minority_pixel = pixel_sorted_by_value[0]
    majority_pixel = pixel_sorted_by_value[-1]
    minority_pixel_value = minority_pixel[0]
    minority_pixel_count = minority_pixel[1]
    majority_pixel_value = majority_pixel[0]
    majority_pixel_count = majority_pixel[1]
    #print("Minority: %s" % minority_pixel_value)
    #print("Minority Count: %s" % minority_pixel_count)
    #print("Majority: %s" % majority_pixel_value)
    #print("Majority Count: %s" % majority_pixel_count)

    pixel_group_count = len(pixel_dict)
    #print("Variety: %s" % pixel_group_count)

    #cv2.imshow('image'+str(count),kpsimage)
    #cv2.imwrite(outfiles[count], kpsimage)

    result_file_lines.append([non_zero, total_pixel_sum, mean_pixel_value, harmonic_mean_pixel_value, pixel_median_value, pixel_variance, pixel_standard_dev, pixel_pstandard_dev, min_pixel, max_pixel, minority_pixel_value, minority_pixel_count, majority_pixel_value, majority_pixel_count, pixel_group_count])

    count += 1


with open(results_outfile, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(result_file_lines)

writeFile.close()

#cv2.waitKey(0)
