from __future__ import absolute_import, print_function
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
import threading
import sys

# Import queue based on python version
if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue

# Import local files
import help_functions
import extract_buildings

#Returns offset for a given map
def getOffset(map):
    x_offset = 0
    y_offset = 0

    if map > 2 and map < 6:
        x_offset = 1
    if map > 6:
        x_offset = 2

    if map == 1 or map == 4 or map == 8:
        y_offset = 1
    if map == 0 or map == 3 or map == 7:
        y_offset = 2

    return x_offset, y_offset

#Returns colors given a certain label
def getColor(l):
    if l < 1:
        return 1, 255, 1  #
    if l < 2:
        return 1, 255, 128  #
    if l < 3:
        return 1, 255, 255  #
    if l < 4:
        return 255, 1, 1  #
    if l < 5:
        return 255, 128, 1  #
    if l < 6:
        return 255, 255, 1  #
    if l < 7:
        return 1, 1, 255  # Red
    if l < 8:
        return 127, 1, 255
    if l < 9:
        return 255, 1, 255
    if l < 10:
        return 255, 173, 106
    if l < 11:
        return 255, 128, 1
    if l < 12:
        return 255, 255, 1
    if l < 13:
        return 1, 1, 153 # Blue
    if l < 14:
        return 1, 1, 255
    if l < 15:
        return 103, 178, 255
    if l < 16:
        return 178, 102, 255
    if l < 17:
        return 255, 1, 255
    if l < 18:
        return 153, 1, 153

    return 1, 1, 1

#Thread worker. Colors all buildings 
def colorer(THREAD_ID, cluster_data, nbr_feat_max, map_c, markers, 
    CORES, cls_mask, nbr_feat_min):
    for feat in range(THREAD_ID, nbr_feat_max, CORES):
        if cluster_data[feat, nbr_feat_min - 1] == map_c:
            if not feat % (nbr_feat_max / 10):
                print("Map: ", map_c, " || Thread: ", THREAD_ID, " || done: ", ((feat * 100) / nbr_feat_max), "%")
            marker_id = cluster_data[feat, nbr_feat_min - 2]
            label = cluster_data[feat, nbr_feat_min]
            r, g, b = getColor(label)
            index_pos = np.where(markers == marker_id)
            cls_mask[index_pos] = [r, g, b]
    if THREAD_ID == 0:
        print("map: ", map_c, " is done!\n\n")

#Colors all buildings. Uses all cores available
def colorCluster(cluster_data, map_source_directory, CORES, 
    scale=None, save=None, im_name='_'):
    if scale is None:
        scale = 0.5

    if save is None:
        save = False

    nbr_feat_min = min(cluster_data.shape) - 1
    nbr_feat_max = max(cluster_data.shape) - 1
    im_size = int(8192 * scale)
    im_full = np.empty([im_size * 3, im_size * 3, 3], dtype=int)

    TIME = time.time()
    print("Coloring Cluster result")
    concatenated = list(range(0, 6)) + list(range(7, 10))
    for map_c in concatenated:
        print("Start coloring map ", map_c)
        # concatenated = chain(range(0, 6),range(7, 10))
        # for map_c in concatenated:

        map_name = map_source_directory[map_c]

        name = "./markers/markers_" + str(map_c) + ".png"
        markers = np.asarray(Image.open(name))
        markers = cv2.resize(markers, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cls_mask = np.empty([max(markers.shape), max(markers.shape), 3], dtype=np.uint8) * 0

        threads = []
        for i in range(0, CORES):
            t = threading.Thread(target=colorer, args=(i, cluster_data, nbr_feat_max, map_c,
                                                       markers, CORES, cls_mask, nbr_feat_min))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        index_pos = np.where(cls_mask[:, :, 0] > 0)
        ort = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
        ort = cv2.resize(ort, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ort[index_pos] = ort[index_pos] * 0.3
        ort = ort + cls_mask * 0.7

        x_offset, y_offset = getOffset(map_c)

        im_full[y_offset * im_size:(y_offset + 1) * im_size,
        x_offset * im_size:(x_offset + 1) * im_size, 0:im_size] = ort

    print("Time:")
    print(time.time() - TIME)

    if save:
        print("Saving Class image")
        print("Shape FD = ", im_full.shape)
        Image.fromarray(im_full.astype('uint8')).save("ColoredImage" + im_name + ".jpeg")
    else:
        print("Show Image")
        Image.fromarray(im_full.astype('uint8')).show()

#Makes a scatterplot of two features
def scatterPlot(cluster_data, feat1, feat2, figure=0, name_x='plot_x',name_y='plot_y'):
    plt.figure(figure)
    plt.plot(cluster_data[:, feat1], cluster_data[:, feat2], 'ro')
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.show()
