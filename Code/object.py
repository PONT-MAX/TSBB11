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

#Returns progress of an operation
def getProgress(x, THREAD_ID, to_range, map_id):
    if not x % (to_range / 10):
        print("Map: ", map_id, "Thread: ", 
            THREAD_ID, ", progress: ", (x * 100 / to_range), "%")


#Returns pixels for patch cutout given a size
def getPixel(indices, cutout_size, image_size):
    row = indices.item(0) - cutout_size / 2
    col = indices.item(1) - cutout_size / 2
    row = max(0, min(row, (image_size[0] - cutout_size)))
    col = max(0, min(col, (image_size[0] - cutout_size)))
    return (row, col)

#Returns patch cutout given a size and the images
def getCutOut(markers, dhm, dtm, cls, row, col, x, cutout_size):
    cutout = np.copy(markers[row:row + cutout_size, col:col + cutout_size])
    dhm_cutout = dhm[row:row + cutout_size, col:col + cutout_size]
    dtm_cutout = dtm[row:row + cutout_size, col:col + cutout_size]
    cls_cutout = cls[row:row + cutout_size, col:col + cutout_size]
    cutout[cutout != x] = 0
    cutout[cutout == x] = 1

    return dhm_cutout, dtm_cutout, cls_cutout, cutout

#Returns some features given the dtm
def getDtmFeatures(dtm_mask):
    sea_level = np.mean(dtm_mask[dtm_mask > 0])
    sea_max = np.amax(dtm_mask)
    ground_slope = (sea_max - sea_level) / sea_max
    return sea_max, ground_slope

#Returns percentage of all classes surrounding current object
def getNeighbourClass(cutout, cls_cutout):
    kernel = np.ones((5, 5), np.uint8)
    cutout_2 = cv2.dilate(np.uint8(cutout), kernel, iterations=20)
    cutout_2 = cutout_2 - np.uint8(cutout)
    aux_mask = cutout_2 * cls_cutout
    sum_of_all = np.count_nonzero(aux_mask)

    terrain = np.sum(aux_mask[aux_mask == 1]) / sum_of_all
    object = np.sum(aux_mask[aux_mask == 2]) / sum_of_all / 2
    forest = np.sum(aux_mask[aux_mask == 3]) / sum_of_all / 3
    water = np.sum(aux_mask[aux_mask == 4]) / sum_of_all / 4
    road = np.sum(aux_mask[aux_mask == 5]) / sum_of_all / 5

    return terrain, forest, road, water, object

#Returns height and area given a mask
def getHeightFeatures(dhm_mask):
    vol = np.sum(dhm_mask) * 0.25  # Normalize for 0.25m^2 ground pixel
    max_height = np.amax(dhm_mask)
    area = np.count_nonzero(dhm_mask) * 0.25  # Normalize for 0.25m^2 ground pixel
    avg_height = vol / area

    # Roof type: 0 = flat, 1 = steep
    #roof_type = (max_height - avg_height) / ((float)(max_height))

    return (max_height, avg_height, area)

#Returns various features given a binary mask
def getFeaturesFromBinary(mark_mask):
    ret, thresh = cv2.threshold(np.uint8(mark_mask), 0, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    cnt = contours[0]
    if max(cnt.shape) > 5:
        good = 1
        (x, y), (width, height), angle = cv2.minAreaRect(cnt)
        contour_area = width * height * 0.25
        contour_ratio = (max(width, height) - min(width, height)) / max(width, height)
        height_out = max(width, height) * 0.5
        width_out = min(width, height) * 0.5
        arc_length = cv2.arcLength(cnt, True)
    else:
        angle = 0
        contour_ratio = 0
        good = 0
        contour_area = 0
        width_out = 0
        height_out = 0
        arc_length = 0

    if angle > 90.0:
        angle = 180 - angle

    return contour_ratio, contour_area, width_out, height_out, good, arc_length

#Saves all markers to PNG files
def saveMarkers(map_source_directory, MIN_BLOB_SIZE, 
    PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2,QUANTIZE_ANGLES):
    for map_c in range(0, 11):
        print("Saving new markers... map: ", map_c)
        map_name = map_source_directory[map_c]
        dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
        cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)

        dhm_norm = np.copy(dhm)
        maxval = np.amax(dhm)
        dhm_norm = dhm_norm / maxval

        object_mask = help_functions.getObject(cls, dhm)
        markers = getMarkers(map_name, map_c, object_mask, dhm_norm,
            MIN_BLOB_SIZE, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES)
        name = "./markers/markers_" + str(map_c) + ".png"
        Image.fromarray(markers).save(name, bits=32)

#Returns markers - numbered binary objects - from a given map
def getMarkers(map_name, map_id, object_mask, dhm_norm, k
    MIN_BLOB_SIZE, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2,QUANTIZE_ANGLES):

    # Import ortho map for watershed. Import house mask.
    ortho = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    _, mask = extract_buildings.getBuildings(ortho, object_mask,dhm_norm, 
        MIN_BLOB_SIZE, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2,QUANTIZE_ANGLES)

    # Finding certain background area
    kernel = np.ones((3, 3), np.uint8)  # Minimal Kernel size.
    dilate_iterations = 3;
    sure_bg = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    # Finding certain foreground area
    sure_fg = np.uint8(mask)

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Watershed
    #Show contours on map, read ortho again to see better
    ortho = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    markers1 = cv2.watershed(ortho, markers)

    return markers1


#Thread worker. Returns features from markers in a map
def extractFeatureData(markers, dhm, dtm, cls, NUMBER_OF_FEATURES,
    map_id, que, CORES, THREAD_ID):
    cutout_size = 1000

    feature_data = np.zeros([NUMBER_OF_FEATURES, 1])
    image_size = dhm.shape
    to_range = np.amax(markers) + 1
    start = 2 + THREAD_ID
    for marker_id in range(start, to_range, CORES):

        # Temporary holder for features
        feature_data_temp = np.empty([NUMBER_OF_FEATURES, 1])

        # Print Progress
        getProgress(marker_id, THREAD_ID, to_range, map_id)

        # Check if object exists
        indices = np.argwhere(markers == marker_id)
        if not np.sum(indices):
            continue

        # Find where the object exists
        row, col = getPixel(indices, cutout_size, image_size)
        
        # Get coutout mask and maps
        dhm_cutout, dtm_cutout, cls_cutout, cutout = getCutOut(markers,
            dhm, dtm, cls, row, col, marker_id, cutout_size)

        # Get features from height map
        dhm_mask = cutout * dhm_cutout
        max_height, avg_height, area = getHeightFeatures(dhm_mask)

        # Check if data is good
        if area < 50.0:  
            continue

        # Get features from binary map
        contour_ratio, contour_area, contour_width, contour_height, good, arc_length = getFeaturesFromBinary(cutout)

        if good == 0:
            #Area too small - not good!
            continue

        # Get percentage of each class neighbouring the object
        # terrain, forest, road, water, object_cls = getNeighbourClass(cutout, cls_cutout)

        # Get data from DTM
        dtm_mask = np.uint8(cutout) * dtm_cutout
        sea_level, ground_slope = getDtmFeatures(dtm_mask)

        feature_data_temp[0, 0] = area  # m^2
        feature_data_temp[1, 0] = max_height  # m
        feature_data_temp[2, 0] = avg_height  # m
        feature_data_temp[3, 0] = sea_level  # m
        feature_data_temp[4, 0] = contour_width  # m
        feature_data_temp[5, 0] = contour_height  # m
        feature_data_temp[6, 0] = arc_length  # m

        feature_data_temp[7, 0] = marker_id
        feature_data_temp[8, 0] = map_id

        feature_data = np.hstack((feature_data, feature_data_temp))

    # Clean and normalize clusterdata
    feature_data = np.delete(feature_data, 0, 1)
    que.put(feature_data)

#Returns features from all maps, uses all cores available.
def getFeatures(map_source_directory, CORES, NUMBER_OF_FEATURES, MIN_BLOB_SIZE,
    PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES, new_markers=None,
    filename=None,load_features=False,save_features=False):

    if new_markers is None:
        new_markers = False
    if filename is None:
        filename = ''

    if new_markers:
        print("Make new markers")
        saveMarkers(map_source_directory, MIN_BLOB_SIZE, 
            PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES)
    elif load_features:
        print("Using old Features")
        return np.transpose(np.load(filename))


    feature_data = np.zeros([NUMBER_OF_FEATURES, 1])
    TIME = time.time()
    for x in range(0, 11):  # 0:11
        # Load Maps
        que = queue.Queue()
        print("Map: ", x)
        map_name = map_source_directory[x]
        dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
        #dtm = cv2.imread('../Data/dtm/' + map_name + 'dtm.tif', -1)
        dsm = cv2.imread('../Data/dsm/' + map_name + 'dsm.tif', -1)
        cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)

        print("Using Old Markers")
        name = "./markers/markers_" + str(x) + ".png"
        markers = np.asarray(Image.open(name))

        print("Map: ", x, "Starting extract Features")

        threads = []
        for i in range(0, CORES):
            t = threading.Thread(target=extractFeatureData,
                                 args=(markers, dhm, dsm, cls, NUMBER_OF_FEATURES, x, que, CORES, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for i in range(0, CORES):
            feature_data = np.hstack((feature_data, que.get()))

        print("After one map done shape: ", feature_data.shape)

    print("Time:")
    print(time.time() - TIME)

    feature_data = np.delete(feature_data, 0, 1)

    if new_markers or save_features:

        print("Saving Features")
        print("Shape FD = ", feature_data.shape)
        print("Are you certain that you changed the number of features...?")
        print("NUMBER_OF_FEATURES= ", NUMBER_OF_FEATURES)
        np.save(filename, feature_data)

    return np.transpose(feature_data)
