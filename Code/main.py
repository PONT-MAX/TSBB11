# import the necessary packages
# from __future__ import print_function
# import numpy as np
# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import sys
import cv2
import numpy as np
import random
# import pyopencl as cl
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

#Import local files
import init_v
import object
import extract_buildings
import help_functions
import multiprocessing


CORES = multiprocessing.cpu_count()
print("Number of System Cores: ", CORES)

#################################

# Load map names
map_source_directory = init_v.init_map_directory()
NUMBER_OF_FEATURES = 13

"""
# Extract features
# Ny mergad funktion
feature_data = object.getFeatures(map_source_directory,CORES,new_markers=False,save=False)




cluster_data = np.transpose(np.load('./numpy_arrays/feature_data_all_threads_final.npy'))
print(cluster_data.shape)


cluster_data_meta = np.empty([max(cluster_data.shape), 3])
cluster_data_meta[:, 0] = np.copy(cluster_data[:, 11]) # Marker id
cluster_data_meta[:, 1] = np.copy(cluster_data[:, 12]) # Map id
cluster_data = np.delete(cluster_data, 11, 1)
cluster_data = np.delete(cluster_data, 11, 1)

cluster_data_first = cluster_data[:,[0, 2]]


print(cluster_data_first.shape)

#best_mcs,best_ms,best_P = object.findOptimalHdbParameters(cluster_data_first,True)

stat = True
print_all_statistic = True
print_mask = True
visulize_clustering = False
best_mcs = 178
best_ms = 119
hd_cluster = object.printOptimalHdb(cluster_data_first,best_mcs,best_ms,stat,print_all_statistic,visulize_clustering)




# Add map number and class to each feature
cluster_data_meta[:, 2] = np.copy(hd_cluster.labels_)
cluster_data = np.hstack((cluster_data, cluster_data_meta)) # 13 49 47

# KLUSTA UNCLASSIFIED DATA!!!!!




print(cluster_data.shape)

cluster_data[:,13] = cluster_data[:,13] + 1

# 5, 80, 2, 1, 2, 2, 6, 40
# 40, 120, 2, 1, 2, 6, 11, 40
for label in range(0, (int)(max(cluster_data[:,13])+1)):
    index_pos = np.where(cluster_data[:, 13] != label)
    index_pos_not = np.where(cluster_data[:, 13] == label)

    cluster_current = np.delete(cluster_data, index_pos, 0)
    cluster_data = np.delete(cluster_data, index_pos_not, 0)
    best_mcs, best_ms, best_P = object.findOptimalHdbParameters(cluster_current[:, 0:11],False)
    print("\nbest_mcs = ", best_mcs, " || ms: ", best_ms, "   %:", best_P)
    hd_cluster = object.printOptimalHdb(cluster_current[:, 0:11], best_mcs, best_ms, False, True,False)

    cluster_current[:, 13] = hd_cluster.labels_

    index_pos = np.where(cluster_current[:, 13] >= 0)
    cluster_current[index_pos, 13] += 100*(label + 1)
    cluster_data = np.vstack((cluster_data, cluster_current))

index_pos = np.where(cluster_data[:, 13] >= 0)
cluster_data[index_pos, 13] -= 100
histo = np.bincount((cluster_data[:,13].astype(int)+1))
print(histo)
print(histo[1:10])
print(histo[101:110])
print(histo[201:210])

np.save('cd1.npy', cluster_data)


"""

cluster_data = np.load('cd1.npy')
# Print out whole map 3x3 w/ scale = 1 or 0.5 or 0,25 or 0.125
# Ny mergad funktion
object.colorCluster(cluster_data, map_source_directory,CORES,scale=0.5)


