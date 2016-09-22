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

map_source_directory = init_v.init_map_directory()
map_name = map_source_directory[5]

# call type w/: dtm,dsm,dhm,cls,ortho
ortho = init_v.get_map_array('ortho', map_name, True)
dtm =   init_v.get_map_array('dtm', map_name, True)
dsm =   init_v.get_map_array('dsm', map_name, True)
dhm =   init_v.get_map_array('dhm', map_name, True)
cls =   init_v.get_map_array('cls', map_name, True)



# Pre processing

# Edge detection adn extraction

# MOG

# Create new map on dtm

# Export to visual


#vei.visulation_export(map_name)
#call(["./visulation/lab"])