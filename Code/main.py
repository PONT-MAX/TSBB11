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
import object
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

#################################

# Load map names

map_source_directory = init_v.init_map_directory()

# Number of features getting extracted, and preparing feature holder
NUMBER_OF_FEATURES = 13
feature_data = np.zeros([NUMBER_OF_FEATURES, 1])
"""

for x in range(0,17):
    # Load Maps
    print("Map: ", x)
    map_name = map_source_directory[x]
    dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
    dsm = cv2.imread('../Data/dsm/' + map_name + 'dsm.tif', -1)
    cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
    image_size = dhm.shape

    print("Map: ", x, "Getting markers")
    # Get markers from map (Watershed) this stage is performed by other function later on
    markers = object.getMarkers(map_name)

    print("Map: ", x, "Starting extract Features")
    feature_data_temp = object.extractFeatureData(markers,dhm,dsm,cls,NUMBER_OF_FEATURES,x)
    feature_data = np.hstack((feature_data, feature_data_temp))


# After all features fro all maps are extracted
# Clean featuredata
feature_data = np.delete(feature_data, 0, 1)
print("Shape FD = ", feature_data.shape)
np.save('./numpy_arrays/feature_data_all_5.npy', feature_data)


"""

cluster_data = np.transpose(np.load('./numpy_arrays/feature_data_all_5.npy'))

cluster_data_meta = np.empty([max(cluster_data.shape), 3])
cluster_data_meta[:, 1] = np.copy(cluster_data[:, 11]) # Marker id
cluster_data_meta[:, 2] = np.copy(cluster_data[:, 12]) # Map id
cluster_data = np.delete(cluster_data, 11, 1)
cluster_data = np.delete(cluster_data, 11, 1)

print(cluster_data.shape)



# print(" Starting HDBSCAN, data size:", cluster_data.shape)
hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=48, min_samples=1, alpha=1.0)
hd_cluster.fit(cluster_data)

# Lables
stat = False
print_all_statistic = False
visulize_clustering = False
proc, nbr_cls = object.printHdbscanResult(hd_cluster, cluster_data, stat, print_all_statistic, visulize_clustering,
                                          48, 1, 5)

"""

#141 1 11
#161 1

for mcs in range(30,60):
    print("MCS: ", mcs)
    best_P = 50
    for ms in range(1, mcs):

        # print(" Starting HDBSCAN, data size:", cluster_data.shape)
        hd_cluster = hdbscan.HDBSCAN(algorithm='best',metric='euclidean',min_cluster_size=mcs,min_samples=ms,alpha=1.0)
        hd_cluster.fit(cluster_data)

        # Lables
        stat = False
        print_all_statistic = False
        visulize_clustering = False
        proc, nbr_cls = object.printHdbscanResult(hd_cluster,cluster_data,stat,print_all_statistic,visulize_clustering,best_P,mcs,ms)

        if nbr_cls > 8 and nbr_cls < 30 and proc < 31:
            print("MCS: ", mcs, " & MS: ", ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
            print_all_statistic = True
            proc, nbr_cls = object.printHdbscanResult(hd_cluster, cluster_data, stat, print_all_statistic,
                                                      visulize_clustering, best_P, mcs, ms)
            best_P = proc

"""


# Add map number and class to each feature
cluster_data_meta[:, 0] = np.copy(hd_cluster.labels_)
cluster_data = np.hstack((cluster_data, cluster_data_meta))
nbr_feat_min = min(cluster_data.shape) - 1
nbr_feat_max = max(cluster_data.shape) - 1
im_size = 2048*2
im_full = np.empty([im_size*3, im_size*3, 3], dtype=int)


le = 0
te = 2
for map_c in (range(0, 3) + range(4, 7) + range(9,12)):
    map_name = map_source_directory[map_c]
    markers = object.getMarkers(map_name)
    cls_mask = np.empty([max(markers.shape),max(markers.shape),3],dtype=np.uint8)
    ort = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    for feat in range(0, nbr_feat_max):
        if cluster_data[feat, nbr_feat_min] == map_c:
            if not feat % 20:
                print("Map: ", map_c, " || ", feat)
            marker_id = cluster_data[feat, nbr_feat_min - 1]
            b, g, r = object.getColor(cluster_data[feat, nbr_feat_min - 2])
            index_pos = np.where(markers == marker_id)
            cls_mask[index_pos] = [r, g, b]

    index_pos = np.where(markers > 2)
    ort[index_pos] = ort[index_pos] * 0.3
    ort = ort + cls_mask*0.7

    res = cv2.resize(ort, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    im_full[te * im_size:(te + 1) * im_size, le * im_size:(le + 1) * im_size, :] = res
    te -= 1
    if map_c == 2 or map_c == 6:
        le += 1
        te = 2


print(im_full.shape)
Image.fromarray(im_full.astype('uint8')).show()

