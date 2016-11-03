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

"""

map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]
dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
dsm = cv2.imread('../Data/dsm/' + map_name + 'dsm.tif', -1)
cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
image_size = dhm.shape
print(image_size)
# For each map
"""
# object.getMarkers(map_name)

#int32
#markers = np.load('./numpy_arrays/markers.npy')
#int32
markers = np.load('./numpy_arrays/markers1.npy')
#int8 3chanels
obj_rgb = np.load('./numpy_arrays/obj_rgb.npy')
"""
indices = np.argwhere(markers==6)
print(indices.item(0))
print(indices.item(1))
markers[markers != 6] = 0
markers[markers == 6] = 255

Image.fromarray(markers).show()

"""
"""
NUMBER_OF_FEATURES = 12
t=time.time()
not_in_Array = 0
cutout_size = 1500

cluster_data = np.zeros([NUMBER_OF_FEATURES,1])

print(cluster_data.shape)

for x in range(2,np.amax(markers)):

    # Temporary holder for fueatures
    cluster_data_temp = np.empty([NUMBER_OF_FEATURES, 1])

    # Print Progress
    object.getProgress(x,20)

    # Check if object exists
    indices = np.argwhere(markers == x)
    if not np.sum(indices):
        continue;


    # NEW FEATURE
    # VART AR objectet GLOBALT?
    # Fraga om orientationproblemet
    # NS till EW
    # Shape rund fyrkantig
    # Avlang eler uniform svd
    # Ihalig eller massiv
    # PCA - uppdelning ?

    # Find where the object exsists
    row,col = object.getPixel(indices,cutout_size,image_size)

    # Get coutout mask and maps
    dhm_cutout, dsm_cutout, cls_cutout, cutout = object.getCutOut(markers,dhm,dsm,cls,row,col,x,cutout_size)

    # Centrum of object
    #cx,cy = object.getArea(cutout)

    # Get features from Height map
    dhm_mask = np.uint8(cutout) * dhm_cutout
    vol,max_height,avg_height,roof_type,area = object.getVolume(dhm_mask)

    # Check if data is good
    if area < 1.0:# or cx > image_size or cy > image_size or cx < 0 or cy < 0:
        print( area)
        continue

    # Get procentage of each class Neighbouring the object
    terrain, forrest, road, water, object_cls = object.getNeighbourClass(cutout, cls_cutout)

    # Get data from DSM
    dsm_mask = np.uint8(cutout) * dsm_cutout
    sea_level, ground_slope = object.getDsmFeatures(dsm_mask)

    # Add data to temporary array
    cluster_data_temp[0, 0] = area
    cluster_data_temp[1, 0] = vol
    cluster_data_temp[2, 0] = max_height
    cluster_data_temp[3, 0] = avg_height
    cluster_data_temp[4, 0] = roof_type
    cluster_data_temp[5, 0] = terrain
    cluster_data_temp[6, 0] = forrest
    cluster_data_temp[7, 0] = road
    cluster_data_temp[8, 0] = water
    cluster_data_temp[9, 0] = object_cls
    cluster_data_temp[10, 0] = ground_slope
    cluster_data_temp[11, 0] = sea_level

    # Get info from DTM (Height, slope)
    # Orientation fit line OpenCV
    # Feature: Not rectangle, circle

    cluster_data = np.hstack((cluster_data, cluster_data_temp))







# Clean and normalize clusterdata
cluster_data = np.delete(cluster_data,0,1)

for x in range(0,NUMBER_OF_FEATURES):
    max_value = np.amax(cluster_data[x, :])
    if max_value > 0:
        cluster_data[x, :] /= max_value
    else:
        cluster_data[x, :] = 0



print("Time:")
print(time.time()-t)
#np.save('./numpy_arrays/cluster_data.npy', cluster_data)


"""

cluster_data = np.transpose(np.load('./numpy_arrays/cluster_data.npy'))
print("data size:")
print (cluster_data.shape)
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
"""

clusterer = hdbscan.HDBSCAN(algorithm='best',metric='euclidean',min_cluster_size=2,min_samples=2,alpha=1.0)
    #HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    #gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    #metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

clusterer.fit(cluster_data)
print(np.amax(clusterer.labels_))


projection = TSNE().fit_transform(cluster_data)

nbr_of_classes = np.amax(clusterer.labels_)

# Lables
histo = np.bincount(clusterer.labels_+1)
print((-1, "-", histo[0]))

color_palette = sns.color_palette('Paired', nbr_of_classes+1)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)


plt.show()




"""


#Image.fromarray(markers).show()

# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])

"""
