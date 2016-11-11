import numpy as np
import cv2
import math
from PIL import Image
import scipy

def getObject(cls,dhm):
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
    return object_mask

def getBlobs(object_mask, ortho ):

    #hough-transform to retrive the buildings
    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(object_mask, 1, np.pi/180, 30, minLineLength, maxLineGap)
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
           cv2.line(ortho, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #converting the ortho-hough image to binary mask
    cv2.imwrite("ortho_png.png", ortho)
    ortho_png=Image.open("ortho_png.png")
    width, height = ortho_png.size

    img=np.zeros((width,height), np.uint8)
    c = (0, 255, 0)
    indices = np.where(np.all(ortho == c, axis=-1))
    img[indices]=255

    #morphology to remove noise/ fill in the blobs
    kernel = np.ones((7,7),np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((10,10),np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)

    #labeling the different blobs and removing the ones smaller than X pixels
    number_of_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(opened,connectivity=4)
    for blob_number in range(0,number_of_blobs):
        if stats[blob_number, 4]<300:
            labels[labels==blob_number]=0

    return number_of_blobs, labels, stats, ortho_png
