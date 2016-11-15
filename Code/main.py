# import the necessary packages
# from __future__ import print_function
# import numpy as np
# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import cv2
import numpy as np
import random
# import pyopencl as cl
import init_v
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
import threading
import Queue


import object
import extract_buildings
import help_functions
import multiprocessing

CORES = multiprocessing.cpu_count()
print("Number of System Cores: ", CORES)

#################################

# Load map names
map_source_directory = init_v.init_map_directory()

# Number of features getting extracted, and preparing feature holder
NUMBER_OF_FEATURES = 13
feature_data = np.zeros([NUMBER_OF_FEATURES, 1])


"""


queue = Queue.Queue()
w0 = threading.Thread(target=object.worker, args=(0,3,map_source_directory,queue)) # 0,1,2
w1 = threading.Thread(target=object.worker, args=(3,5,map_source_directory,queue)) # 3,4
w2 = threading.Thread(target=object.worker, args=(5,8,map_source_directory,queue)) # 5,6,7
w3 = threading.Thread(target=object.worker, args=(8,11,map_source_directory,queue)) # 8, 9,10

w0.start()
w1.start()
w2.start()
w3.start()

print(" After Start!!")

w0.join()
w1.join()
w2.join()
w3.join()

print(" After Join!!")

feature_data_0 = queue.get()
feature_data_1 = queue.get()
feature_data_2 = queue.get()
feature_data_3 = queue.get()

print(" After get!!")


print(feature_data_0.shape)
print("F0: \n", feature_data_0[0:2,0:3])
print(feature_data_1.shape)
print("F1: \n", feature_data_1[0:2,0:3])
print(feature_data_2.shape)
print("F2: \n", feature_data_2[0:2,0:3])
print(feature_data_3.shape)
print("F3: \n", feature_data_3[0:2,0:3])

print("Is done!!")

feature_data = np.hstack((feature_data, feature_data_0))
feature_data = np.hstack((feature_data, feature_data_1))
feature_data = np.hstack((feature_data, feature_data_2))
feature_data = np.hstack((feature_data, feature_data_3))

# After all features fro all maps are extracted
# Clean featuredata
feature_data = np.delete(feature_data, 0, 1)
print("Shape FD = ", feature_data.shape)
np.save('./numpy_arrays/feature_data_all_threads_final.npy', feature_data)

"""
"""
cluster_data = np.transpose(np.load('./numpy_arrays/feature_data_all_threads_final.npy'))
"""

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
for label in xrange(0, (int)(max(cluster_data[:,13])+1)):
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
print(cluster_data.shape)
queue_color = Queue.Queue()

w0 = threading.Thread(target=object.colorer, args=(0, 3, map_source_directory, cluster_data, queue_color)) # 0,1,2
w1 = threading.Thread(target=object.colorer, args=(3, 6, map_source_directory, cluster_data, queue_color)) # 3,4,5
w2 = threading.Thread(target=object.colorer, args=(7, 10, map_source_directory, cluster_data, queue_color)) # 7,8,9
# 2, 5, 9

# 5,7
# 8,9
#

w0.start()
w1.start()
w2.start()

print(" After Start!!")

w0.join()
w1.join()
w2.join()

print(" After Join!!")

im_labeled_0 = queue_color.get()
im_labeled_1 = queue_color.get()
im_labeled_2 = queue_color.get()

Image.fromarray(im_labeled_0.astype('uint8')).show()
Image.fromarray(im_labeled_1.astype('uint8')).show()
Image.fromarray(im_labeled_2.astype('uint8')).show()


# Merge Image

if im_labeled_0[0,0,0] == 0:
    tot_im = im_labeled_0
elif im_labeled_1[0,0,0] == 0:
    tot_im = im_labeled_1
else:
    tot_im = im_labeled_2

if im_labeled_0[0,0,0] == 3:
    tot_im = np.hstack((tot_im, im_labeled_0))
elif im_labeled_1[0,0,0] == 3:
    tot_im = np.hstack((tot_im, im_labeled_1))
else:
    tot_im = np.hstack((tot_im, im_labeled_2))

if im_labeled_0[0,0,0] == 7:
    tot_im = np.hstack((tot_im, im_labeled_0))
elif im_labeled_1[0,0,0] == 7:
    tot_im = np.hstack((tot_im, im_labeled_1))
else:
    tot_im = np.hstack((tot_im, im_labeled_2))

Image.fromarray(tot_im.astype('uint8')).show()


"""

