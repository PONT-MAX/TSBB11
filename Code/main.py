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
import sub_clustering as sc


#Import local files
import init_v
import object
import extract_buildings
import help_functions
import multiprocessing
import cluster

#Set constants
CORES = multiprocessing.cpu_count()
print("Number of System Cores: ", CORES, "\n")
NUMBER_OF_FEATURES = 9

#########################################
# Load map names
map_source_directory = init_v.init_map_directory()

# Extract features
print("Loading feature data... ")
feature_data = object.getFeatures(map_source_directory,
    CORES,new_markers=False,filename='./numpy_arrays/feature_data_all_threads_final.npy', load=True)



data = sc.kMeansSubCluster(feature_data,normalize=True)
object.colorCluster(data,map_source_directory,CORES,save=True,im_name='New_data', scale=0.25)
#sc.exportCluster2PNG(data, map_source_directory, CORES)


