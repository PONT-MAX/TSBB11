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

import object
import extract_buildings
import help_functions

#################################

# Load map names
map_source_directory = init_v.init_map_directory()

# Number of features getting extracted, and preparing feature holder
NUMBER_OF_FEATURES = 13
feature_data = np.zeros([NUMBER_OF_FEATURES, 1])

#Extract features from a selected amount of maps or use an existing numpy array.
create_features = False
first_map = 0
last_map = 11 #11 in total
filename = './numpy_arrays/feature_data_all.npy'
cluster_data = object.getAllFeatures(first_map,last_map,filename,create_features)

#Meta cluster data for subclustering.
cluster_data_meta = np.empty([max(cluster_data.shape), 3])
cluster_data_meta[:, 0] = np.copy(cluster_data[:, 11]) # Marker id
cluster_data_meta[:, 1] = np.copy(cluster_data[:, 12]) # Map id
cluster_data = np.delete(cluster_data, 11, 1)
cluster_data = np.delete(cluster_data, 11, 1)

cluster_data_first = cluster_data[:,[0, 2]]


print(cluster_data_first.shape)

#best_mcs,best_ms,best_P = object.findOptimalHdbParameters(cluster_data,True)
best_mcs = 16
best_ms = 3
stat = True
print_all_statistic = True
print_mask = True
visulize_clustering = False

hd_cluster = object.printOptimalHdb(cluster_data,best_mcs,best_ms,stat,print_all_statistic,visulize_clustering)




# Add map number and class to each feature
cluster_data_meta[:, 2] = np.copy(hd_cluster.labels_)
cluster_data = np.hstack((cluster_data, cluster_data_meta)) # 13
   
"""


print(cluster_data.shape)
index_pos = np.where(cluster_data[:, 13] < 0)
index_pos_del = np.where(cluster_data[:, 13] > -1)
cluster_data_un_cls = np.delete(cluster_data, index_pos_del, 0)
cluster_data = np.delete(cluster_data, index_pos, 0)

# 5, 80, 2, 1, 2, 2, 6, 40
# 40, 120, 2, 1, 2, 6, 11, 40
for label in xrange(0, (int)(max(cluster_data[:,13])+1)):
    index_pos = np.where(cluster_data[:, 13] != label)
    index_pos_not = np.where(cluster_data[:, 13] == label)

    cluster_current = np.delete(cluster_data, index_pos, 0)
    cluster_data = np.delete(cluster_data, index_pos_not, 0)
    best_mcs, best_ms, best_P = object.findOptimalHdbParameters(cluster_current[:, 0:11],False)
    print("\nbest_mcs = ", best_mcs, " || ms: ", best_ms, "   %:", best_P)
    hd_cluster = object.printOptimalHdb(cluster_current, best_mcs, best_ms, False, True,False)

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



cluster_data = np.load('cd1.npy')
print(cluster_data.shape)
"""
nbr_feat_min = min(cluster_data.shape) - 1
nbr_feat_max = max(cluster_data.shape) - 1
im_size = 2048*2
im_full = np.empty([im_size*3, im_size*3, 3], dtype=int)

le = 0
te = 2
concatenated = list(range(0, 6)) + list(range(7,10))
for map_c in concatenated:

#concatenated = chain(range(0, 6),range(7, 10))
#for map_c in concatenated:
    print(map_c)
    map_name = map_source_directory[map_c]
    dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
    cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
    object_mask = help_functions.getObject(cls,dhm)
    #markers = np.load('markerstmp.npy')
    markers = object.getMarkers(map_name,map_c, object_mask)
    #np.save('markerstmp.npy', markers)
    #input('save done')
    cls_mask = np.empty([max(markers.shape),max(markers.shape),3],dtype=np.uint8)
    ort = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    for feat in range(0, nbr_feat_max):
        
        if cluster_data[feat, nbr_feat_min-1] == map_c:
            if not feat % 20:
                print("Map: ", map_c, " || ", feat)
            marker_id = cluster_data[feat, nbr_feat_min - 2]
            label = cluster_data[feat, nbr_feat_min]
            if label == -1:
                continue
            b, g, r = object.getColor(label)
            index_pos = np.where(markers == marker_id)
            cls_mask[index_pos] = [r, g, b]

    index_pos = np.where(markers > 2)
    ort[index_pos] = ort[index_pos] * 0.3
    ort = ort + cls_mask*0.7

    res = cv2.resize(ort, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    im_full[te * im_size:(te + 1) * im_size, le * im_size:(le + 1) * im_size, :] = res
    te -= 1
    if map_c == 2 or map_c == 5:
        le += 1
        te = 2


print(im_full.shape)
Image.fromarray(im_full.astype('uint8')).show()

"""
"""
