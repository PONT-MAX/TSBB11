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
#from matplotlib import pyplot as plt


map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]
dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
# For each map

# object.getMarkers(map_name)

#int32
#markers = np.load('./numpy_arrays/markers.npy')
#int32
markers = np.load('./numpy_arrays/markers1.npy')
#int8 3chanels
obj_rgb = np.load('./numpy_arrays/obj_rgb.npy')

markers[markers != 800] = 0
markers[markers > 0] = 254

cx,cy,area = object.getArea(markers)

markers[markers > 0] = 1
dhm_mask = np.float32(markers)*dhm

vol,max_height,avg_height,roof_type = object.getVolume(dhm_mask,area)





#Image.fromarray(markers).show()

# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])

