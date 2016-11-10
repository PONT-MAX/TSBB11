from __future__ import absolute_import, print_function
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
import hdbscan


def getCorrectGlobalMapPosition(map):
    x = 0
    y = 0
    im_size = 8192
    if map == 0 or map == 4 or map == 9 or map == 14:
        y = 4*im_size
    if map == 1 or map == 5 or map == 10 or map == 15:
        y = 3*im_size
    if map == 2 or map == 6 or map == 11 or map == 16:
        y = 2*im_size
    if map == 3 or map == 7 or map == 12:
        y = 2*im_size

    if map > 3 and map < 9:
        x = im_size
    if map > 8 and map < 14:
        x = 2*im_size
    if map > 13 and map < 17:
        x = 3*im_size

    return (x,y)


def getProgress(x,delay):
    if not x % delay:
        print(x)



def getPixel(indices,cutout_size,image_size):
    row = indices.item(0) - cutout_size / 2
    col = indices.item(1) - cutout_size / 2
    row = max(0, min(row, (image_size[0] - cutout_size)))
    col = max(0, min(col, (image_size[0] - cutout_size)))
    return (row,col)

def getCutOut(markers,dhm,dsm,cls,row,col,x,cutout_size):
    cutout = np.copy(markers[row:row + cutout_size, col:col + cutout_size])
    dhm_cutout = np.uint8(dhm[row:row + cutout_size, col:col + cutout_size])
    dsm_cutout = np.uint8(dsm[row:row + cutout_size, col:col + cutout_size])
    cls_cutout = cls[row:row + cutout_size, col:col + cutout_size]

    # TO BE DONE: Extract classdata and orthodata
    # ortho_cutout = np.copy(ortho[row:row + cutout_size, col:col + cutout_size])

    cutout[cutout != x] = 0
    cutout[cutout == x] = 1

    return dhm_cutout,dsm_cutout, cls_cutout, cutout

def getDsmFeatures(dsm_mask):

    sea_level = np.mean(dsm_mask[dsm_mask > 0])
    sea_max = np.amax(dsm_mask)
    ground_slope = (sea_max-sea_level)/sea_max
    return sea_max, ground_slope


def getNeighbourClass(cutout,cls_cutout):
    # Return procentage of all classes surounding the current object

    kernel = np.ones((5, 5), np.uint8)
    cutout_2 = cv2.dilate(np.uint8(cutout),kernel,  iterations=5)
    cutout_2 = cutout_2 - np.uint8(cutout)
    aux_mask = cutout_2*cls_cutout
    sum_of_all = np.count_nonzero(aux_mask)

    terrain    = np.sum(aux_mask[aux_mask == 1])/sum_of_all
    object_cls = np.sum(aux_mask[aux_mask == 2])/sum_of_all/2

    if np.sum(aux_mask[aux_mask == 4]) > 0:
        water      = 1.0
    else:
        water = 0.0

    road       = np.sum(aux_mask[aux_mask == 5])/sum_of_all/5
    forest     = np.sum(aux_mask[aux_mask == 3])/sum_of_all/3

    return terrain,forest,road,water,object_cls

def getVolume(dhm_mask):
    vol = np.sum(dhm_mask)*0.25 # Normalize for 0.25m^2 ground pixel
    max_height = np.amax(dhm_mask)
    area = np.count_nonzero(dhm_mask)*0.25 # Normalize for 0.25m^2 ground pixel
    avg_height = vol / area

    #Do func gets roof type
    # res 0   = platt
    # res 0.5 = sluttande
    # res 1.0 = extremt sluttande (typ kyrka)

    roof_type = (max_height - avg_height)/((float)(max_height))

    if False:
        print("Vol=  ")
        print(vol)
        print("Max Height=  ")
        print(max_height)
        print("Average height = ")
        print(avg_height)
        print("Roof type")
        print (roof_type)


    return (max_height,avg_height,area)

def getArea(mark_mask):
    ret, thresh = cv2.threshold(np.uint8(mark_mask), 0, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cnt = contours[0]
    if max(cnt.shape) > 5:
        (x, y), (MA, ma), angle =  cv2.fitEllipse(cnt)
        contour_ratio = (max(MA,ma) - min(MA,ma))/max(MA,ma)
        good = 1
    else:
        angle = 0
        contour_ratio = 0
        good = 0
        print("CNT FALSE!!")

    if angle > 90.0:
        angle = 180 - angle

    if False:
        print("pos: x,y = ")
        print(x)
        print(y)
        print("Rotation: theta = ")
        print(angle)
        print("MA = ", MA )
        print("ma = ", ma)
        print(contour_ratio)

    return contour_ratio, good

def getMarkers(map_name):
    # call type w/: dtm,dsm,dhm,cls,ortho
    dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
    cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
    # extract tall bouldings
    # less then 2 meters high is not an object (house)
    dhm[dhm < 1.5] = 0
    dhm_mask = np.copy(dhm)
    dhm_mask[dhm_mask > 0] = 1
    cls[cls != 2] = 0
    cls[cls > 0] = 1

    obj_mask = cls * np.uint8(dhm_mask)
    # Put to 255 for show
    obj_mask[obj_mask > 0] = 1
    #Image.fromarray(obj_mask*255).show()
    obj_mask_med = cv2.medianBlur(obj_mask, 5)
    #Image.fromarray(obj_mask_med*255).show()

    # dhm_obj = ((dhm*obj_mask_med)/np.amax(dhm))*255.0
    dhm_obj = (dhm * obj_mask_med)
    #Image.fromarray(dhm_obj*255).show()


    obj = np.uint8(dhm_obj)

    # Plot histogram
    # plt.hist(obj.ravel(),256,[0,256])
    # plt.show()

    # blur = cv2.GaussianBlur(obj,(1,1),0)
    med = cv2.medianBlur(obj, 9)

    # ret, thresh = cv2.threshold(med,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(med, 4, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(med,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,0)

    #Image.fromarray(obj*255).show('obj')
    #Image.fromarray(thresh).show('threshold')



    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # Tweeka itterations
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    # Tweeka itreataions
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FIXA THRESHOLD

    #Image.fromarray(sure_fg).show('foreground')
    #Image.fromarray(sure_bg).show('background')
    #Image.fromarray(unknown).show('unknown')
    #Image.fromarray(dist_transform).show('dist')


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Watershed need colorimage
    obj_rgb = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)

    # Watershed
    markers1 = cv2.watershed(obj_rgb, markers)

    #obj_rgb[markers1 == -1] = [255, 255, 0]
    #Image.fromarray(obj_rgb).show()

    #Image.fromarray(markers1).show()

    # Binary data
    # np.save('./numpy_arrays/markers.npy', markers)
    # np.save('./numpy_arrays/markers1.npy', markers1)
    # np.save('./numpy_arrays/obj_rgb.npy', obj_rgb)

    return markers1

def printHdbscanResult(hd_cluster,feature_data,stat,print_all,visulize_clustering,best,mcs,ms):

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
        sum_all_p = sum_all*100/nbr_of_datapoints
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

def extractFeatureData(markers,dhm,dsm,cls,NUMBER_OF_FEATURES,map_id):
   
    t = time.time()
    cutout_size = 1500

    feature_data = np.zeros([NUMBER_OF_FEATURES, 1])
    image_size = dhm.shape
    to_range = np.amax(markers) + 1

    for marker_id in range(2, to_range):

        # Temporary holder for fueatures
        feature_data_temp = np.empty([NUMBER_OF_FEATURES, 1])

        # Print Progress
        getProgress(marker_id, 50)

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
            #print(area)
            continue

        # Centrum of object
        contour_ratio, good = getArea(cutout)

        if good == 0:
            print("Not good")
            continue

        # Get procentage of each class Neighbouring the object
        terrain, forrest, road, water, object_cls = getNeighbourClass(cutout, cls_cutout)

        # Get data from DSM
        dsm_mask = np.uint8(cutout) * dsm_cutout
        sea_level, ground_slope = getDsmFeatures(dsm_mask)


        # Add data to temporary array
        feature_data_temp[0, 0] = area # m^2
        feature_data_temp[1, 0] = max_height
        feature_data_temp[2, 0] = avg_height
        feature_data_temp[3, 0] = terrain
        feature_data_temp[4, 0] = forrest
        feature_data_temp[5, 0] = road
        feature_data_temp[6, 0] = water
        feature_data_temp[7, 0] = object_cls
        feature_data_temp[8, 0] = ground_slope
        feature_data_temp[9, 0] = sea_level
        feature_data_temp[10, 0] = contour_ratio
        feature_data_temp[11, 0] = marker_id
        feature_data_temp[12, 0] = map_id

        feature_data = np.hstack((feature_data, feature_data_temp))

    # Clean and normalize clusterdata
    feature_data = np.delete(feature_data, 0, 1)

    print("Time:")
    print(time.time() - t)
    # np.save('./numpy_arrays/feature_data_16.npy', feature_data)
    
    return feature_data


def getColor(class_nbr):
    b = 0
    g = 0
    r = 0
    sat = 255

    #if class_nbr == -1 or class_nbr == 1 or class_nbr == 2 or class_nbr == 4 or class_nbr == 7 or class_nbr > 9:
        #return b,g,r

    if class_nbr == 0:
        r = sat
    elif class_nbr == 1:
        g = sat
    elif class_nbr == 2:
        b = sat
    elif class_nbr == 3:
        b = sat
        g = sat
    elif class_nbr == 4:
        b = sat
        r = sat
    elif class_nbr == 5:
        g = sat
        r = sat
    elif class_nbr == 6:
        b = sat
        r = sat
        g = sat

    sat = 120
    if class_nbr == 7:
        b = sat
    elif class_nbr == 8:
        g = sat
    elif class_nbr == 9:
        r = sat
    elif class_nbr == 10:
        b = sat
        g = sat
    elif class_nbr == 11:
        b = sat
        r = sat
    elif class_nbr == 12:
        g = sat
        r = sat
    elif class_nbr == 13:
        b = sat
        r = sat
        g = sat
    elif class_nbr > 13:
        b = 255
        r = 100
        g = 200
    elif class_nbr == -1:
        b = 55
        r = 55
        g = 55

    return (b, g, r)

def findOptimalHdbParameters(cluster_data):
    print("Finding optimal parameters for HDBSCAN with for-loops, data size:", cluster_data.shape)
    mcs_start = int(input("Enter starting minimum cluster size: "))
    mcs_end = int(input("Enter maximum minimum cluster size: "))
    mcs_delta = int(input("Enter for-loop counter increment: "))
    ms_start = int(input("Enter starting minimum samples: "))
    ms_delta = int(input("Enter for-loop counter increment: "))
    nbr_cls_low = int(input("Enter lowest number of classes: "))
    nbr_cls_high = int(input("Enter highest number of classes: "))
    proc_high = int(input("Enter lowest outlier percentage: "))
    best_mcs = 0
    best_ms = 0
    best_P = 50
    #prompt user for mcs, nbr_cls, proc
    for mcs in range(mcs_start,mcs_end,mcs_delta):
        print("MCS: ", mcs)
        
        for ms in range(1, mcs, ms_delta):

            # print(" Starting HDBSCAN, data size:", cluster_data.shape)
            hd_cluster = hdbscan.HDBSCAN(algorithm='best',metric='euclidean',min_cluster_size=mcs,min_samples=ms,alpha=1.0)
            hd_cluster.fit(cluster_data)

            # Lables
            stat = False
            print_all_statistic = False
            visulize_clustering = False
            proc, nbr_cls = printHdbscanResult(hd_cluster,cluster_data,stat,print_all_statistic,visulize_clustering,best_P,mcs,ms)

            if nbr_cls > nbr_cls_low and nbr_cls < nbr_cls_high and proc < proc_high:
                print("MCS: ", mcs, " & MS: ", ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
                best_mcs = mcs
                best_ms = ms
                best_P = proc
                
    return (best_mcs,best_ms,best_P)


def printOptimalHdb(cluster_data,mcs, ms, stat, print_all_statistic,visulize_clustering):

    hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=mcs, min_samples=ms, alpha=1.0)
    hd_cluster.fit(cluster_data)

    # Lables
    proc, nbr_cls = printHdbscanResult(hd_cluster, cluster_data, stat, print_all_statistic, visulize_clustering,
                                              141, 1, 5)
    return hd_cluster


