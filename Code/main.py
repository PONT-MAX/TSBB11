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
import visulation_export_image as vei
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
from itertools import chain

#Import local files
import init_v
import object
import extract_buildings
import help_functions
import multiprocessing
import cluster


CORES = multiprocessing.cpu_count()
print("Number of System Cores: ", CORES)

#################################

# Load map names
map_source_directory = init_v.init_map_directory()
NUMBER_OF_FEATURES = 13

save_cluster_data = False
save_filename = 'cd1.npy'



cluster_data = cluster.cluster_data(map_source_directory,
    save_cluster_data,save_filename,CORES)

object.colorCluster(cluster_data, map_source_directory,CORES,scale=0.5,save=True)