# import the necessary packages
# from __future__ import print_function
# import numpy as np
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import cv2
import numpy as np
import random
#import pyopencl as cl
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
NUMBER_OF_FEATURES = 17
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
    #feature_data_temp = object.extractFeatureData(markers,dhm,dsm,cls,NUMBER_OF_FEATURES,x)
    #feature_data = np.hstack((feature_data, feature_data_temp))


# After all features fro all maps are extracted
# Clean featuredata
feature_data = np.delete(feature_data, 0, 1)
print("Shape FD = ", feature_data.shape)
#np.save('./numpy_arrays/feature_data_all_4.npy', feature_data)

"""

cluster_data = np.transpose(np.load('./numpy_arrays/feature_data_all_4.npy'))


cluster_data_meta = np.empty([max(cluster_data.shape), 4])
cluster_data_meta[:, 0] = np.copy(cluster_data[:,16])
cluster_data_meta[:, 2] = np.copy(cluster_data[:,12])
cluster_data_meta[:, 3] = np.copy(cluster_data[:,13])
cluster_data = np.delete(cluster_data, 16, 1)
cluster_data = np.delete(cluster_data, 12, 1)
cluster_data = np.delete(cluster_data, 12, 1)

print(max(cluster_data[:,4]))
print(max(cluster_data[:,2]))
#for x in range(0,min(cluster_data.shape)):
 #   print(cluster_data[4000:4020, x])
  #  cluster_data[:,x] = cluster_data[:,x]/max(cluster_data[:,x])
   # print(cluster_data[4000:4020, x])

print(cluster_data.shape)


#best_mcs,best_ms,best_P = object.findOptimalHdbParameters(cluster_data)
best_mcs = 29
best_ms = 8

stat = True
print_all_statistic = True
visulize_clustering = False
hd_cluster = object.printOptimalHdb(cluster_data,best_mcs,best_ms,stat,print_all_statistic,visulize_clustering)

# Add map number and class to each feature
cluster_data_meta[:,1] = np.copy(hd_cluster.labels_)
cluster_data = np.hstack((cluster_data, cluster_data_meta))

for map_c in range(5,6):

    map_name = map_source_directory[map_c]
    ort = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', -1)/2

    xc,yc = object.getCorrectGlobalMapPosition(map_c)
    print(xc,yc)
    c = 0
    for feat in range(0,8210):
        if cluster_data[feat,14-5] == map_c:
            c += 1
            x = (int)(cluster_data[feat, 17-5] - xc)
            y = (int)(cluster_data[feat, 16-5] - yc)
            b,g,r = object.getColor(cluster_data[feat, 15-5])
            cv2.rectangle(ort, (x, y), (x+50, y+50), (b, g, r), 2)
            #print("x = ", x, "  y = ", y, "   ||| xc,yc = ", xc, yc)


    Image.fromarray(ort.astype('uint8')).show()




"""


plt.figure(1)
plt.plot(cluster_data[0, :], cluster_data[2, :], 'ro')
plt.ylabel('area vs max h')

plt.figure(2)
plt.plot(cluster_data[2, :], cluster_data[4, :], 'ro')
plt.ylabel('height vs type')

plt.figure(3)
plt.plot(cluster_data[0, :], cluster_data[11, :], 'ro')
plt.ylabel('vol vs dsm')

plt.show()

#vei.visulation_export(map_name)
#call(["./visulation/lab"])
"""

