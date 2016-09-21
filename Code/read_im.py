# import the necessary packages
# from __future__ import print_function
#import cv2
# import numpy as np
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
from numpy import *
import pyopencl as cl
from PIL import Image

map_name = '0153811e_582244n_20160905T073406Z_'
"""
# Data
ortho = array(Image.open('../Data/ortho/' + map_name + 'tex.tif'))
ortho_out = Image.fromarray(ortho.astype(uint8))
#ortho_out.show()
ortho_out.save('visulation/images/ortho.bmp')




dsm = array(Image.open('../Data/dsm/' + map_name + 'dsm.tif'))
dsm[dsm < 0] = 0
dsm_n = (dsm/dsm.max())*255.0
dsm_out = Image.fromarray(dsm.astype(uint8))
dsm_out_n = Image.fromarray(dsm_n.astype(uint8))
dsm_out.save('visulation/images/dsm.bmp')
dsm_out_n.save('visulation/images/dsm_n.bmp')



dhm = array(Image.open('../Data/dhm/' + map_name + 'dhm.tif'))
dhm[dhm < 0] = 0
dhm_n = (dhm/dhm.max())*255.0
dhm_out = Image.fromarray(dhm.astype(uint8))
dhm_out_n = Image.fromarray(dhm_n.astype(uint8))
dhm_out.save('visulation/images/dhm.bmp')
dhm_out_n.save('visulation/images/dhm_n.bmp')


dtm = array(Image.open('../Data/dtm/' + map_name + 'dtm.tif'))
dtm[dtm < 0] = 0
dtm_n = (dtm/dtm.max())*255.0
dtm_out = Image.fromarray(dtm.astype(uint8))
dtm_out_n = Image.fromarray(dtm_n.astype(uint8))
dtm_out.save('visulation/images/dtm.bmp')
dtm_out_n.save('visulation/images/dtm_n.bmp')

"""

aux = array(Image.open('../Data/auxfiles/' + map_name + 'cls.tif'))
aux_out = Image.fromarray(aux.astype(uint8))
aux_out.save('visulation/images/aux.bmp')
