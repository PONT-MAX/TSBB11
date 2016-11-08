# import the necessary packages
# from __future__ import print_function
# import numpy as np
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
#import pyopencl as cl
import init_v
import visulation_export_image as vei
from subprocess import call
from PIL import Image
import cv2
import math
import extract_buildings

map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]


# call type w/: dtm,dsm,dhm,cls,ortho
dtm =   init_v.get_map_array('dtm', map_name, True)
dhm =   init_v.get_map_array('dhm', map_name, True)
cls =   init_v.get_map_array('cls', map_name, True)
ortho =   init_v.get_map_array('ortho', map_name, True)



# Pre processing

# Extract treshold
# less then 2 meters high is not an object (house)
dhm[dhm<2] = 0

# Make copy for use later
cls2 = np.copy(cls)
# Remove obejct class
cls2[cls2 == 2] = 0

# Extract object class
cls[cls != 2] = 0
cls[cls == 2] = 1

object_mask = np.multiply(dhm,cls)
object_mask[object_mask>0.0] = 2
cls2 = cls2 + object_mask



lines=extract_buildings.get_buildings(ortho, object_mask)


# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])


Image.fromarray(lines).show()

#cv2.imshow('hough', lines)
#cv2.waitKey(0)

