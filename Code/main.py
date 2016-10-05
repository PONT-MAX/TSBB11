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



map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]
dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
print(cls.dtype)
image_size = dhm.shape
print(image_size)
# For each map

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

t=time.time()
not_in_Array = 0
cutout_size = 1500

cluster_data = np.zeros([10,1])

print(cluster_data.shape)

for x in range(2,np.amax(markers)):

    cluster_data_temp = np.empty([10, 1])

    if not x%20:
        print(x)
        print(not_in_Array)

    indices = np.argwhere(markers == x)

    if not np.sum(indices):
        not_in_Array += 1
        print("NOT")
        print(x)
        continue;

    row = indices.item(0) - cutout_size/2
    col = indices.item(1) - cutout_size/2
    row = max(0,min(row,(image_size[0] - cutout_size)))
    col = max(0,min(col,(image_size[0] - cutout_size)))

    cutout = np.copy(markers[row:row+cutout_size,col:col+cutout_size])
    dhm_cutout = np.copy(np.uint8(dhm[row:row+cutout_size,col:col+cutout_size]))
    cls_cutout = np.copy(cls[row:row + cutout_size, col:col + cutout_size])

    #TO BE DONE: Extract classdata and orthodata
    #ortho_cutout = np.copy(ortho[row:row + cutout_size, col:col + cutout_size])

    cutout[cutout != x] = 0
    cutout[cutout == x] = 1


    #cx,cy = object.getArea(cutout)


    dhm_mask = np.uint8(cutout)*dhm_cutout
    vol,max_height,avg_height,roof_type,area = object.getVolume(dhm_mask)

    if area < 1.0:# or cx > image_size or cy > image_size or cx < 0 or cy < 0:
        print( area)
        continue

    cluster_data_temp[0,0] = area
    cluster_data_temp[1,0] = vol
    cluster_data_temp[2,0] = max_height
    cluster_data_temp[3,0] = avg_height
    cluster_data_temp[4,0] = roof_type
    #most reapeted haighet most repeted


    kernel = np.ones((5, 5), np.uint8)
    cutout_2 = cv2.dilate(np.uint8(cutout),kernel,  iterations=3)
    cutout_2 = cutout_2 - np.uint8(cutout)
    cls_mask = cutout_2*cls_cutout
    terrain, forrest, road, water, object_cls = object.getNeighbourClass(cls_mask)

    cluster_data_temp[5, 0] = terrain
    cluster_data_temp[6, 0] = forrest
    cluster_data_temp[7, 0] = road
    cluster_data_temp[8, 0] = water
    cluster_data_temp[9, 0] = object_cls


    cluster_data = np.hstack((cluster_data, cluster_data_temp))



cluster_data = np.delete(cluster_data,0,1)
print("Time:")
print(time.time()-t)
#np.save('./numpy_arrays/cluster_data.npy', cluster_data)

"""

cluster_data = np.load('./numpy_arrays/cluster_data.npy')
print("data size:")
print (cluster_data.shape)

plt.figure(1)
plt.plot(cluster_data[0,:],cluster_data[8,:],'ro')
plt.ylabel('area vs water')

plt.figure(2)
plt.plot(cluster_data[2,:],cluster_data[4,:],'ro')
plt.ylabel('height vs type')

plt.figure(3)
plt.plot(cluster_data[0,:],cluster_data[9,:],'ro')
plt.ylabel('vol vs cls')

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