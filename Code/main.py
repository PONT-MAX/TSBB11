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
NUMBER_OF_FEATURES = 15

#########################################
# Load map names
map_source_directory = init_v.init_map_directory()
date = datetime.datetime.now()
feature_data_filename = './numpy_arrays/feature_data_all_threads_final_' + \
	str(date.month) + str(date.day) + str(date.hour) + str(date.minute) + '.npy'
feature_data_filename ='./numpy_arrays/feature_data_all_threads_final_1211423.npy'

# Extract features
print("Loading feature data... ")
feature_data = object.getFeatures(map_source_directory, CORES, NUMBER_OF_FEATURES,
	new_markers=False,filename=feature_data_filename,
	load_features=True,save_features=False)


print("\n")
#feature_data[:,1] = np.square(feature_data[:,1])
feature_data = feature_data[:, np.array([0,1,2,8,9,10,11,12,13,14])]

#cluster_data1 = cluster.cluster_data(feature_data,
#    save_cluster_data=True,save_filename='cd_full_cluster1.npy',sub_clustering=False)





cluster_data2 = cluster.cluster_data(feature_data,
    save_cluster_data=True,save_filename='cd_sub_cluster1.npy',sub_clustering=True)






print("\n")
object.colorCluster(cluster_data2, map_source_directory,CORES,scale=0.5,save=False,sub_c=True)
#object.colorCluster(cluster_data1, map_source_directory,CORES,scale=0.5,save=False,sub_c=False)
