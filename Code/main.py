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
#from matplotlib import pyplot as plt


map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]
dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
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


print("max")
print(np.amax(markers))
t=time.time()
not_in_Array = 0
cutout_size = 1500

for x in range(2,100):

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

    #TO BE DONE: Extract classdata and orthodata
    #cls_cutout = np.copy(cls[row:row + cutout_size, col:col + cutout_size])
    #ortho_cutout = np.copy(ortho[row:row + cutout_size, col:col + cutout_size])

    cutout[cutout != x] = 0
    cutout[cutout == x] = 1
    cx,cy,area = object.getArea(cutout)
    dhm_mask = np.uint8(cutout)*dhm_cutout
    vol,max_height,avg_height,roof_type = object.getVolume(dhm_mask,area)

print("Time:")
print(time.time()-t)
print("Not in arr:")
print(not_in_Array)




"""


#Image.fromarray(markers).show()

# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])

"""