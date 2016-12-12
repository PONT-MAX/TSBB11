#Import libraries

from __future__ import absolute_import, print_function
import cv2
import numpy as np
import random
import visulation_export_image as vei
from subprocess import call
from PIL import Image
from PIL import ImageOps
import time
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from itertools import chain
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

#Import local files
import init_v
import object
import extract_buildings
import help_functions
import multiprocessing
import cluster
import sub_clustering
import hdbscan_clustering
import visualization

#Set constants
CORES = multiprocessing.cpu_count()
MIN_BLOB_SIZE = 300
PERCENTAGEOFARC1 = 0.01
PERCENTAGEOFARC2 = 0.005
NUMBER_OF_FEATURES = 9
#Needs to be changed if additional features are added 
QUANTIZE_ANGLES = True
#Makes angles straighter
#########################################

print("Number of System Cores: ", CORES, "\n")

# Load map names
map_source_directory = init_v.init_map_directory()
"""
date = datetime.datetime.now()
#feature_data_filename = './numpy_arrays/feature_data_all_threads_final_' + \
	#str(date.month) + str(date.day) + str(date.hour) + str(date.minute) + '.npy'
feature_data_filename ='./numpy_arrays/feature_data_all_threads_final.npy'

# Extract features
print("Loading feature data... ")
feature_data = object.getFeatures(map_source_directory, CORES, NUMBER_OF_FEATURES,
 MIN_BLOB_SIZE, PERCENTAGEOFARC1, PERCENTAGEOFARC2, QUANTIZE_ANGLES, new_markers=True, 
 filename=feature_data_filename,load_features=False,save_features=True)
print("Done!")
print("Clustering data... ")
data = sub_clustering.kMeansSubCluster(feature_data, normalize=True)
print("Done!")
print("Coloring data... ")
visualization.colorCluster(data, map_source_directory, CORES, save=True, 
	im_name='New_data2', scale=0.125)
sub_clustering.exportCluster2PNG(data, map_source_directory, CORES)
print("Done!")

"""
print("Print monopoly houses...")
help_functions.printMonopolyHouses(map_source_directory, MIN_BLOB_SIZE, PERCENTAGEOFARC1,
	PERCENTAGEOFARC2, QUANTIZE_ANGLES)
print("Done!")

