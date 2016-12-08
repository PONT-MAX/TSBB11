from __future__ import absolute_import, print_function
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
import hdbscan
import threading
import sys

#Import queue based on python version
if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue

#Import local files
import help_functions
import extract_buildings

def getCorrectGlobalMapPosition(map):
    x = 0
    y = 0
    im_size = 8192
    if map == 0 or map == 4 or map == 9 or map == 14:
        y = 4 * im_size
    if map == 1 or map == 5 or map == 10 or map == 15:
        y = 3 * im_size
    if map == 2 or map == 6 or map == 11 or map == 16:
        y = 2 * im_size
    if map == 3 or map == 7 or map == 12:
        y = 2 * im_size

    if map > 3 and map < 9:
        x = im_size
    if map > 8 and map < 14:
        x = 2 * im_size
    if map > 13 and map < 17:
        x = 3 * im_size

    return (x, y)

def getProgress(x, delay,THREAD_ID,to_range,map_id):
    if not x % (to_range/10):
        print("Map: ", map_id, "Thread: ", THREAD_ID, ", progress: ", (x*100/to_range), "%")

def getPixel(indices, cutout_size, image_size):
    row = indices.item(0) - cutout_size / 2
    col = indices.item(1) - cutout_size / 2
    row = max(0, min(row, (image_size[0] - cutout_size)))
    col = max(0, min(col, (image_size[0] - cutout_size)))
    return (row, col)

def getCutOut(markers, dhm, dsm, cls, row, col, x, cutout_size):
    cutout = np.copy(markers[row:row + cutout_size, col:col + cutout_size])
    dhm_cutout = np.uint8(dhm[row:row + cutout_size, col:col + cutout_size])
    dsm_cutout = np.uint8(dsm[row:row + cutout_size, col:col + cutout_size])
    cls_cutout = cls[row:row + cutout_size, col:col + cutout_size]

    # TO BE DONE: Extract classdata and orthodata
    # ortho_cutout = np.copy(ortho[row:row + cutout_size, col:col + cutout_size])

    cutout[cutout != x] = 0
    cutout[cutout == x] = 1

    return dhm_cutout, dsm_cutout, cls_cutout, cutout

def getDsmFeatures(dsm_mask):
    sea_level = np.mean(dsm_mask[dsm_mask > 0])
    sea_max = np.amax(dsm_mask)
    ground_slope = (sea_max - sea_level) / sea_max
    return sea_max, ground_slope

def getNeighbourClass(cutout, cls_cutout):
    # Return procentage of all classes surounding the current object

    kernel = np.ones((5, 5), np.uint8)
    cutout_2 = cv2.dilate(np.uint8(cutout), kernel, iterations=10)
    cutout_2 = cutout_2 - np.uint8(cutout)
    aux_mask = cutout_2 * cls_cutout
    sum_of_all = np.count_nonzero(aux_mask)

    terrain = np.sum(aux_mask[aux_mask == 1]) / sum_of_all
    object_cls = np.sum(aux_mask[aux_mask == 2]) / sum_of_all / 2

    if np.sum(aux_mask[aux_mask == 4]) > 0:
        water = 1.0
    else:
        water = 0.0

    road = np.sum(aux_mask[aux_mask == 5]) / sum_of_all / 5
    forest = np.sum(aux_mask[aux_mask == 3]) / sum_of_all / 3

    return terrain, forest, road, water, object_cls

def getVolume(dhm_mask):
    vol = np.sum(dhm_mask) * 0.25  # Normalize for 0.25m^2 ground pixel
    max_height = np.amax(dhm_mask)
    area = np.count_nonzero(dhm_mask) * 0.25  # Normalize for 0.25m^2 ground pixel
    avg_height = vol / area

    # Do func gets roof type
    # res 0   = platt
    # res 0.5 = sluttande
    # res 1.0 = extremt sluttande (typ kyrka)

    roof_type = (max_height - avg_height) / ((float)(max_height))

    if False:
        print("Vol=  ")
        print(vol)
        print("Max Height=  ")
        print(max_height)
        print("Average height = ")
        print(avg_height)
        print("Roof type")
        print(roof_type)

    return (max_height, avg_height, area)

def getArea(mark_mask):
    ret, thresh = cv2.threshold(np.uint8(mark_mask), 0, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    cnt = contours[0]
    if max(cnt.shape) > 5:
        #(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        #contour_ratio = (max(MA, ma) - min(MA, ma)) / max(MA, ma)
        good = 1
        (x, y), (width, height), angle = cv2.minAreaRect(cnt)
        contour_area = width*height*0.25
        contour_ratio = (max(width, height) - min(width, height)) / max(width, height)
        contour_perimeter = cv2.arcLength(cnt,closed=False) #All buildings are closed contours.
    else:
        angle = 0
        contour_ratio = 0
        good = 0
        contour_area = 0
        contour_perimeter = 0
        print("CNT FALSE!!")

    if angle > 90.0:
        angle = 180 - angle

    if False:
        print("pos: x,y = ")
        print(x)
        print(y)
        print("Rotation: theta = ")
        print(angle)

    return contour_ratio, contour_area, good, contour_perimeter


def saveMarkers(map_source_directory):
    for map_c in range(0, 11):
        print("Saving new markers... map: ", map_c)
        map_name = map_source_directory[map_c]
        dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
        cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)

        dhm_norm = np.copy(dhm)
        maxval = np.amax(dhm)
        dhm_norm = dhm_norm / maxval

        object_mask = help_functions.getObject(cls, dhm)
        markers = getMarkers(map_name, map_c, object_mask, dhm_norm)
        name = "./markers/markers_" + str(map_c) + ".png"
        Image.fromarray(markers).save(name, bits=32)

def getMarkers(map_name, map_id, object_mask,dhm_norm):
    # TODO: tweak iterations for sure bg and fg.
    # TODO: should sure_fg be the input mask or should we erode it?

    # Import ortho map for watershed. Import house mask.
    ortho = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    _, mask = extract_buildings.getBuildings(ortho, object_mask,dhm_norm)

    # Finding certain background area
    kernel = np.ones((3, 3), np.uint8)  # Minimal Kernel size.
    dilate_iterations = 3;
    sure_bg = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    # Finding certain foreground area
    """
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thresh_limit = 0.075 #Most likely 0.7
    ret, sure_fg = cv2.threshold(dist_transform, thresh_limit * dist_transform.max(), 255, 0)
    #sure_fg = np.uint8(sure_fg)
    """
    sure_fg = np.uint8(mask)

    # Finding unknown regionq   <qq
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Image.fromarray(sure_fg).show('foreground')
    # input("Showing FG. Press enter to continue...")
    # Image.fromarray(sure_bg).show('background')
    # input("Showing BG. Press enter to continue...")
    """
    Image.fromarray(unknown).show('unknown')
    Image.fromarray(dist_transform).show('dist')
    """
    # Marker labeling

    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Watershed
    print("Show contours on map, read ortho again to see better")
    ortho = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
    markers1 = cv2.watershed(ortho, markers)
    ortho[markers1 == -1] = [255, 255, 0]
    # Image.fromarray(ortho).show() #SHOWS CONTOURS OF ALL OBJECTS IN ENTIRE MAP
    # input("Press enter to continue...")

    # Image.fromarray(markers1).show()

    # Binary data
    # np.save('./numpy_arrays/markers.npy', markers)
    # np.save('./numpy_arrays/markers1.npy', markers1)
    # np.save('./numpy_arrays/obj_rgb.npy', obj_rgb)

    return markers1

def printHdbscanResult(hd_cluster, feature_data, stat=True, print_all=False, visulize_clustering=False):
    histo = np.bincount(hd_cluster.labels_ + 1)
    nbr_of_datapoints = max(feature_data.shape)
    not_classified = histo[0]
    nbr_of_classes = max(histo.shape)
    proc = not_classified * 100 / nbr_of_datapoints

    if stat:
        print("number of classes(including not a class): ", nbr_of_classes)
        print("Of ", nbr_of_datapoints, " datapoints, ", not_classified, " was not classified.  ", proc, "%")

    if print_all:
        sum_all = 0
        print("class    members     %")
        for x in range(1, nbr_of_classes):
            proc = histo[x] * 100 / nbr_of_datapoints
            sum_all += histo[x]
            print(x, "     ", histo[x], "       ", proc)
        sum_all_p = sum_all * 100 / nbr_of_datapoints
        print("sum    ", sum_all, "      ", sum_all_p)

    if visulize_clustering:
        color_palette = sns.color_palette('Paired', nbr_of_classes + 1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in hd_cluster.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, hd_cluster.probabilities_)]
        projection = TSNE().fit_transform(feature_data)
        plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
        plt.show()

    return (proc, nbr_of_classes)

def getColor(l,sub_cluster=False):


    if not sub_cluster:
        return getColorFull(l)

    if l < 1:
        return 255, 255, 255
    if l < 2:
        return 255, 1,   1
    if l < 3:
        return 1,   255, 1
    if l < 4:
        return 1,   1,   255
    if l < 101:
        return 255, 255, 1
    if l < 102:
        return 255, 1,   255
    if l < 103:
        return 1,   255, 255
    if l < 104:
        return 255, 128, 1
    if l < 201:
        return 153, 255, 51
    if l < 202:
        return 1,   153, 76
    if l < 203:
        return 102, 178, 255
    if l < 204:
        return 127, 1,   255
    if l < 301:
        return 255, 153, 204
    if l < 302:
        return 250, 158, 200
    if l < 303:
        return 128, 1,   1
    if l < 304:
        return 1,   128, 1
    if l < 401:
        return 1,   1,   128
    if l < 402:
        return 128, 128, 1


def getColorFull(l):

    if l < 1:
        return 255, 153, 204
    if l < 2:
        return 255, 1,   1
    if l < 3:
        return 1,   255, 1
    if l < 4:
        return 1,   1,   255
    if l < 5:
        return 255, 255, 1
    if l < 6:
        return 255, 1,   255
    if l < 7:
        return 1,   255, 255
    if l < 8:
        return 255, 128, 1
    if l < 9:
        return 153, 255, 51
    if l < 10:
        return 1,   153, 76
    if l < 11:
        return 102, 178, 255
    if l < 12:
        return 127, 1,   255
    if l < 13:
        return 255, 153, 204
    if l < 14:
        return 250, 235, 240
    if l < 15:
        return 128, 1,   1
    if l < 16:
        return 1,   128, 1
    if l < 17:
        return 1,   1,   128
    if l < 18:
        return 128, 128, 1



    return 1,1,1


def findOptimalHdbParameters(cluster_data, save=False, mcs_start=4,mcs_end=400, mcs_delta=1, ms_delta=1,
                             cls_low=4, cls_high=5, proc=40):

    if max(cluster_data.shape) < 2500:
        mcs_start = 4
        mcs_end = 100


    if save is False:
        best_parameters = np.load("./HdbP/HdbParameters_12461.npy")
        return best_parameters.item(0), best_parameters.item(1)


    print("Sub Clustering")
    mcs_end = min(mcs_end, int(max(cluster_data.shape)/2))


    best_mcs, best_ms, best_P = findClusterParameters(cluster_data, mcs_start, mcs_end, mcs_delta,
                                                      ms_delta, cls_low, cls_high, proc)


    best_parameters = np.array([best_mcs, best_ms, best_P])
    print(best_parameters)
    if save:
        name = "./HdbP/HdbParameters_" + str(max(cluster_data.shape)) + ".npy"
        np.save(name, best_parameters)

    print(best_parameters)
    return best_parameters.item(0), best_parameters.item(1)


def findClusterParameters(cluster_data, mcs_start, mcs_end, mcs_delta, ms_delta, nbr_cls_low,
                          nbr_cls_high, proc_high):

    best_mcs = 0
    best_ms = 0
    best_P = 90



    for mcs in range(mcs_start, mcs_end, mcs_delta):
        good_param_found = False
        if not mcs % 20:
            print("MCS: ", mcs)
            #ms_delta += 1
        for ms in range(1, mcs, ms_delta):
            # print(" Starting HDBSCAN, data size:", cluster_data.shape)
            hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=mcs, min_samples=ms,
                                         alpha=1.0)
            hd_cluster.fit(cluster_data)

            # Lables
            proc, nbr_cls = printHdbscanResult(hd_cluster, cluster_data, stat=False)

            if nbr_cls >= nbr_cls_low and nbr_cls < nbr_cls_high and proc < proc_high:
                # print("MCS: ", mcs, " & MS: ", ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
                if proc < best_P:
                    best_mcs = mcs
                    best_ms = ms
                    best_P = proc
                    print("MCS: ", best_mcs, " & MS: ", best_ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
                    printHdbscanResult(hd_cluster, cluster_data, print_all=True)
                    if best_P < 3:
                        print("Breaking search: Proc < 5%")
                        good_param_found = True
                        break
        if good_param_found:
            break

    return int(best_mcs), int(best_ms), int(best_P)


def printOptimalHdb(cluster_data, mcs, ms, stat=True, print_all_statistic=True, visulize_clustering=False):
    print("optimal: ", mcs, ms)


    hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=mcs, min_samples=ms,
                                 alpha=1.0)
    hd_cluster.fit(cluster_data)

    # Lables
    proc, nbr_cls = printHdbscanResult(hd_cluster, cluster_data,
                                       stat, print_all_statistic, visulize_clustering)
    return hd_cluster

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

def colorer(TREAHD_ID, cluster_data, nbr_feat_max, map_c, markers, CORES, cls_mask, nbr_feat_min,sub=False):

    for feat in range(TREAHD_ID, nbr_feat_max, CORES):
        if cluster_data[feat, nbr_feat_min - 1] == map_c:
            if not feat % (nbr_feat_max / 10):
                print("Map: ", map_c, " || Thread: ", TREAHD_ID, " || done: ", ((feat * 100) / nbr_feat_max), "%")
            marker_id = cluster_data[feat, nbr_feat_min - 2]
            label = cluster_data[feat, nbr_feat_min]
            b, g, r = getColor(label,sub_cluster=sub)
            index_pos = np.where(markers == marker_id)
            cls_mask[index_pos] = [r, g, b]
    if TREAHD_ID == 0:
        print("map: ", map_c, " is done!\n\n")

def colorCluster(cluster_data, map_source_directory, CORES, scale=None,save=None,sub_c=True):
    if scale is None:
        scale = 0.5

    if save is None:
        save = False

    if sub_c:
        print("Coloring Subcluster Style")
    else:
        print("Coloring Full cluster")


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
                                                       markers, CORES, cls_mask, nbr_feat_min,sub_c))
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

        im_full[y_offset * im_size:(y_offset + 1) * im_size, x_offset * im_size:(x_offset + 1) * im_size,
        0:im_size] = ort

    print("Time:")
    print(time.time() - TIME)



    if save:
        print("Saving Class image")
        print("Shape FD = ", im_full.shape)
        Image.fromarray(im_full.astype('uint8')).save("ColoredImage.jpeg")
    else:
        print("Show Image")
        Image.fromarray(im_full.astype('uint8')).show()


def extractFeatureData(markers, dhm, dsm, cls, NUMBER_OF_FEATURES, map_id,que,CORES,THREAD_ID):

    cutout_size = 1000

    feature_data = np.zeros([NUMBER_OF_FEATURES, 1])
    image_size = dhm.shape
    to_range = np.amax(markers) + 1
    start = 2 + THREAD_ID
    for marker_id in range(start, to_range,CORES):

        # Temporary holder for fueatures
        feature_data_temp = np.empty([NUMBER_OF_FEATURES, 1])

        # Print Progress
        getProgress(marker_id, 50,THREAD_ID,to_range,map_id)

        # Check if object exists
        indices = np.argwhere(markers == marker_id)
        if not np.sum(indices):
            continue

        # Find where the object exsists
        row, col = getPixel(indices, cutout_size, image_size)
        # Get coutout mask and maps
        dhm_cutout, dsm_cutout, cls_cutout, cutout = getCutOut(markers, dhm, dsm, cls, row, col, marker_id, cutout_size)

        # Get features from Height map
        dhm_mask = np.uint8(cutout) * dhm_cutout
        max_height, avg_height, area = getVolume(dhm_mask)

        # Check if data is good
        if area < 50.0:  # or cx > image_size or cy > image_size or cx < 0 or cy < 0:
            # print(area)
            continue

        # Centrum of object
        contour_ratio, contour_area, good, contour_perimeter = getArea(cutout)

        if good == 0:
            print("Not good")
            continue

        # Get procentage of each class Neighbouring the object
        terrain, forrest, road, water, object_cls = getNeighbourClass(cutout, cls_cutout)

        # Get data from DSM
        dsm_mask = np.uint8(cutout) * dsm_cutout
        sea_level, ground_slope = getDsmFeatures(dsm_mask)

        # Add data to temporary array
        feature_data_temp[0, 0] = area  # m^2
        feature_data_temp[1, 0] = max_height # m
        feature_data_temp[2, 0] = avg_height # m
        feature_data_temp[3, 0] = terrain
        feature_data_temp[4, 0] = forrest
        feature_data_temp[5, 0] = road
        feature_data_temp[6, 0] = water
        feature_data_temp[7, 0] = object_cls
        feature_data_temp[8, 0] = ground_slope
        feature_data_temp[9, 0] = sea_level # m
        feature_data_temp[10, 0] = contour_ratio # add w / h (m)
        feature_data_temp[11, 0] = min(1, (area/contour_area)) #
        feature_data_temp[12, 0] = contour_perimeter #
        feature_data_temp[13, 0] = marker_id
        feature_data_temp[14, 0] = map_id

        feature_data = np.hstack((feature_data, feature_data_temp))

    # Clean and normalize clusterdata
    feature_data = np.delete(feature_data, 0, 1)


    print("T_ID: ",THREAD_ID, " shape: ", feature_data.shape)
    que.put(feature_data)

def getFeatures(map_source_directory,CORES,NUMBER_OF_FEATURES,new_markers=None,
    filename=None,load_features=False,save_features=False):
    """thread worker function"""
    
    if new_markers is None:
        new_markers = False
    if filename is None:
        filename = ''

    if new_markers:
        print("Make new markers")
        saveMarkers(map_source_directory)
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
        dsm = cv2.imread('../Data/dsm/' + map_name + 'dsm.tif', -1)
        cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)


        print("Using Old Markers")
        name = "./markers/markers_" + str(x) + ".png"
        markers = np.asarray(Image.open(name))

        print("Map: ", x, "Starting extract Features")

        threads = []
        for i in range(0, CORES):
            t = threading.Thread(target=extractFeatureData,
                                 args=(markers, dhm, dsm, cls, NUMBER_OF_FEATURES, x,que,CORES,i))
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
        np.save(filename, feature_data)

    return feature_data
