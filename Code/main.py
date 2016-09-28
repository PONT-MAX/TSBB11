# import the necessary packages
# from __future__ import print_function
# import numpy as np
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import cv2
import numpy as np
#import pyopencl as cl
import init_v
import visulation_export_image as vei
from subprocess import call
from PIL import Image
from PIL import ImageOps


map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]

# For each map


# call type w/: dtm,dsm,dhm,cls,ortho
dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif',-1)
cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif',0)
print(np.amax(cls))
# extract tall bouldings
# less then 2 meters high is not an object (house)
dhm[dhm<4.9] = 0
dhm_mask = np.copy(dhm)
dhm_mask[dhm_mask>0] = 1
cls[cls != 2] = 0
cls[cls > 0] = 1

obj_mask =  np.copy(cls)
obj_mask = cls*np.uint8(dhm_mask)
# Put to 255 for show
obj_mask[obj_mask>0] = 1
#Image.fromarray(obj_mask).show()
obj_mask_med = cv2.medianBlur(obj_mask,21)
#Image.fromarray(obj_mask_med).show()

dhm_obj = dhm*obj_mask_med
#Image.fromarray(dhm_obj).show()


obj = np.uint8(dhm_obj)
ret, thresh = cv2.threshold(obj,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
# Tweeka itterations
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

# sure background area
#Tweeka itreataions
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255,0)



# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

Image.fromarray(sure_fg).show('foreground')
#Image.fromarray(sure_bg).show('background')
#Image.fromarray(unknown).show('unknown')
#Image.fromarray(dist_transform).show('dist')


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

markers[unknown==255] = 0

# Now, mark the region of unknown with zero
#markers[unknown==255] = 0

#Image.fromarray(markers).show()

print("markers")
print(markers.shape)
print(markers.dtype)
print("obj")
print(obj.shape)
print(obj.dtype)

obj_rgb  = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
print("obj_rgb")
print(obj_rgb.shape)
print(obj_rgb.dtype)

markers1 = cv2.watershed(obj_rgb,markers)
obj_rgb[markers1 == -1] = [255,0,0]

Image.fromarray(obj_rgb).show()


"""

res = cv2.resize(sure_fg,None,fx=0.12, fy=0.12, interpolation = cv2.INTER_CUBIC)
cv2.imshow('sure fore ground',res)
cv2.waitKey(0)

res = cv2.resize(sure_bg,None,fx=0.12, fy=0.12, interpolation = cv2.INTER_CUBIC)
cv2.imshow('sure back ground',res)
cv2.waitKey(0)

res = cv2.resize(dist_transform,None,fx=0.12, fy=0.12, interpolation = cv2.INTER_CUBIC)
cv2.imshow('dist_transform',res)
cv2.waitKey(0)
cv2.destroyAllWindows()



# extract tall bouldings
# less then 2 meters high is not an object (house)
dhm_np[dhm_np<2] = 0

# Extract object class
cls_np[cls_np != 2] = 0
cls_np[cls_np == 2] = 1
dhm_obj_np = np.multiply(dhm_np,cls_np)
print(cls_np.dtype)

dhm_obj_cv = cv2.cvtColor(dhm_obj_np, cv2.COLOR_GRAY)
res = cv2.resize(dhm_obj_cv,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""



# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])