import numpy as np
from PIL import Image
import cv2

def getVolume(dhm_mask,area):
    vol = np.sum(dhm_mask)
    max_height = np.max(dhm_mask)
    avg_height = vol / area

    #Do func gets roof type
    # res 0   = platt
    # res 0.5 = sluttande
    # res 1.0 = extremt sluttande (typ kyrka)

    roof_type = (max_height-avg_height)/max_height

    if False:
        print("Vol=  ")
        print(vol)
        print("Max Height=  ")
        print(max_height)
        print("Average height = ")
        print(avg_height)
        print("Roof type")
        print (roof_type)


    return (vol,max_height,avg_height,roof_type)

def getArea(mark_mask):
    ret, thresh = cv2.threshold(np.uint8(mark_mask), 0, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)


    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = cv2.contourArea(cnt)

    if False:
        print("pos: x,y = ")
        print(cx)
        print(cy)
        print("Area")
        print(area)

    return (cx,cy,area)

def getMarkers(map_name):
    # call type w/: dtm,dsm,dhm,cls,ortho
    dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
    cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
    print(np.amax(cls))
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
    # Image.fromarray(obj_mask).show()
    obj_mask_med = cv2.medianBlur(obj_mask, 5)
    # Image.fromarray(obj_mask_med).show()

    # dhm_obj = ((dhm*obj_mask_med)/np.amax(dhm))*255.0
    dhm_obj = (dhm * obj_mask_med)
    # Image.fromarray(dhm_obj).show()


    obj = np.uint8(dhm_obj)

    # Plot histogram
    # plt.hist(obj.ravel(),256,[0,256])
    # plt.show()

    # blur = cv2.GaussianBlur(obj,(1,1),0)
    med = cv2.medianBlur(obj, 9)

    # ret, thresh = cv2.threshold(med,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(med, 4, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(med,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,0)

    # Image.fromarray(obj).show('obj')
    # Image.fromarray(thresh).show('threshold')


    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # Tweeka itterations
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    # Tweeka itreataions
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.20 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FIXA THRESHOLD

    # Image.fromarray(sure_fg).show('foreground')
    # Image.fromarray(sure_bg).show('background')
    # Image.fromarray(unknown).show('unknown')
    # Image.fromarray(dist_transform).show('dist')


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
    obj_rgb[markers1 == -1] = [255, 255, 0]

    #Image.fromarray(obj_rgb).show()
    #Image.fromarray(markers1).show()

    # Binary data
    np.save('./numpy_arrays/markers.npy', markers)
    np.save('./numpy_arrays/markers1.npy', markers1)
    np.save('./numpy_arrays/obj_rgb.npy', obj_rgb)

    return