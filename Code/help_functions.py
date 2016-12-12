import numpy as np
import cv2
import math
from PIL import Image
import scipy
import object
import extract_buildings

#Returns thresholded dhm on a certain elevation.
def getObject(cls,dhm):
    # Less then 2 meters high is not an object (house)
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

#Finds the buildings in the ortho-map and returns them in a binary mask
    #Returns the number of blobs, a labeled "binary" mask and information of each blob
def getBlobs(object_mask, ortho, minblob):

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
        if stats[blob_number, 4] < minblob:
            labels[labels==blob_number]=0

    return number_of_blobs, labels, stats, ortho_png


#Crops out a part of the image
   #Returns the patch and the corner points in the initial image
def cropImPart(ortho, stats,margin):
    (h,w) = ortho.shape[:2]

    left_coord = stats[0]
    top_coord = stats[1]
    right_coord = stats[2] + left_coord
    bottom_coord = stats[3] + top_coord

    if left_coord > margin:
        left_coord = stats[0] - margin
    else:
        left_coord = 0

    if top_coord > margin:
        top_coord = stats[1] - margin
    else:
        top_coord = 0

    if right_coord + margin > w:
        right_coord = w
    else:
        right_coord = right_coord + margin

    if bottom_coord + margin > h:
        bottom_coord = h
    else:
        bottom_coord = bottom_coord + margin

    cropped_im = ortho[top_coord:bottom_coord, left_coord:right_coord]

    return cropped_im, [left_coord, right_coord, top_coord, bottom_coord]

#Returns a binary image with the rotated bounding boxes filled in
def getRotatedBox(bin_im):
    #start at 1 since 0 is background
    bin_im_out=bin_im.copy()
    ret, contours, hierarchy = cv2.findContours(bin_im, 
        mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE )
    length=len(contours)-1
    feat =np.zeros((length,3))
    for cnt in range(0, length):
        rect = cv2.minAreaRect(contours[cnt+1])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(bin_im_out, [box], 0, (255, 0, 0), 2)

    return bin_im_out

#A function that rotates buildings a little in order to align them.
def quantizeAngle(rect):
        (width,height)=rect[1]
        angle=rect[2]
        angle = math.floor(angle)
        quote = width/height
        angle = angle + 180
        #if quote > 1:
        if 8 <= angle <= 22:
            angle = 15
        if 23 <= angle <= 37:
            angle = 30
        if 38 <= angle <= 52:
            angle = 45
        if 53 <= angle <= 67:
            angle = 60
        if 68 <= angle <= 82:
            angle = 75
        if 83 <= angle <= 97:
            angle = 90
        if 98 <= angle <= 112:
            angle = 105
        if 113 <= angle <= 127:
            angle = 120
        if 128 <= angle <= 142:
            angle = 135
        if 143 <= angle <= 157:
            angle = 150
        if 158 <= angle <= 172:
            angle = 165
        if 173 <= angle or angle <= 7:
            angle = 0
        recta = (rect[0],rect[1], angle)
        return recta

#Returns a mask with an approximation of the box shapes
def getApproxBoxes(bin_im, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, quantize):
    bin_im_out = bin_im.copy()
    ret, contours, hierarchy = cv2.findContours(bin_im, 
        mode = cv2.RETR_EXTERNAL, method =cv2.CHAIN_APPROX_SIMPLE)
    bin_im_out = cv2.merge((bin_im_out,bin_im_out,bin_im_out)) # Making bin_im_out 3 channel to display colors.
    for cnt in contours:
            (x,y), (w,h), angle = cv2.minAreaRect(cnt)
            rectarea = w*h
            area = cv2.contourArea(cnt)
            fillArea = area/rectarea
            # Forces bounding boxes onto ish rectangular buildings
            if fillArea > 0.65 and area <8000:
                rect = cv2.minAreaRect(cnt)

                if quantize:
                    newRect=quantizeAngle(rect)
                    box = cv2.boxPoints(newRect)
                else:
                    box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(bin_im_out,[box],0,(255,0,0),-1)
            # Draw bigger objects with help from perimeter lenght
            else:
                epsilon = PERCENTAGE_OF_ARC1 * cv2.arcLength(cnt,True)
                area_cont = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                area_approx = cv2.contourArea(approx)
                fillArea2 = area_approx/area_cont
                cv2.drawContours(bin_im_out,[approx],0,(0,0,255),-1)
                if fillArea2<0.95 and area>10000:
                    epsilon = PERCENTAGE_OF_ARC2 * cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    cv2.drawContours(bin_im_out,[approx],0,(0,0,255),-1)

    # Make the red and blue contours into white
    width, height, _ = bin_im_out.shape
    img=np.zeros((width, height), np.uint8)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    indices = np.where(np.all(bin_im_out == red, axis=-1))
    img[indices]=255
    img2=np.zeros((width, height), np.uint8)
    indices2 = np.where(np.all(bin_im_out == blue, axis=-1))
    img2[indices2]=255
    bin_im_out = img + img2

    return bin_im_out

#Separating buildings in the large blobs into smaller blobs
def blobSep(im, dhm):
    #separating buildings in the large blobs into smaller blobs
    im=im/255
    object_mask = np.multiply(dhm,im)
    limit=np.mean(dhm)
    object_mask=object_mask>limit
    object_mask=object_mask*255
    object_mask=np.uint8(object_mask)
    return object_mask

#Finds the very large buildings and replaces them with a separated version
def moreMorph(bin_im, labeled, no_blobs, stats, meanval, dhm):

    #limi=2*meanval
    limi=1.6*meanval
    large_areas=[]
    for blob in range(1, no_blobs):
        if stats[blob,4]>limi:
            large_areas.append(blob)
    for large in large_areas:
        patch,_=cropImPart(bin_im, stats[large, :],0)
        dhm_patch,_=cropImPart(dhm, stats[large, :],0)
        separated=blobSep(patch, dhm_patch)
        bin_im[stats[large, 1]:(stats[large, 1]+stats[large, 3]), 
            stats[large, 0]:(stats[large, 0]+stats[large, 2])] = separated

    return bin_im

def printMonopolyHouses(map_source_directory,minBlob, PERCENTAGE_OF_ARC1, 
    PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES):
    for map_c in range(0, 11):
        map_name = map_source_directory[map_c]
        ortho = cv2.imread('../Data/ortho/' + map_name + 'tex.tif', 1)
        cls = cv2.imread('../Data/auxfiles/' + map_name + 'cls.tif', 0)
        dhm = cv2.imread('../Data/dhm/' + map_name + 'dhm.tif', -1)
        dtm = cv2.imread('../Data/dtm/' + map_name + 'dtm.tif', -1)
        dhm_modified = np.copy(dhm)
        dtm_modified = np.copy(dtm)
        object_mask = getObject(cls,dhm)

        _, binary_mask = extract_buildings.getBuildings(ortho,object_mask,
            dhm,minBlob,PERCENTAGE_OF_ARC1,PERCENTAGE_OF_ARC2,QUANTIZE_ANGLES)

        number_of_blobs, labels,_, _ = cv2.connectedComponentsWithStats(binary_mask,connectivity=4)
        _, normalized_mask = cv2.threshold(binary_mask, 0, 1, cv2.THRESH_BINARY)
    
        for blob_number in range(0,number_of_blobs):
            vals_from_mask = normalized_mask[labels == blob_number]
            if vals_from_mask[0] > 0:
                median_dhm = np.median(dhm[labels == blob_number])
                median_dtm = np.median(dtm[labels == blob_number])
                dtm_modified[labels == blob_number] = median_dtm
                dhm_modified[labels == blob_number] = median_dhm
    
        houses = np.multiply(dhm_modified, normalized_mask)
        good_height = houses + dtm_modified
        filename = '../Data/monopoly/' + map_name + "monopoly.png"
        Image.fromarray(good_height.astype('uint8')).save(filename)

#Legacy function - takes a line in an image as starting and endpoint
#   returns a list with all points on the line
def lineIter(img, line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        t = x1
        x1 = y1
        y1 = t

        t = x2
        x2 = y2
        y2 = t
    also_steep = x1 > x2
    if also_steep:

        t = x1
        x1 = x2
        x2 = t

        t = y1
        y1 = y2
        y2 = t

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0; # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy/dx)

    if y1 < y2:
        y_step = 1 
    else:
        y_step = -1

    y = y1
    ret = list([])
    for x in range(x1, x2 + 1):
        if steep:
            p = (y, x)
        else:
            p = (x, y)
        if p[0] <= img.shape[1] and p[1] <= img.shape[0]:
            ret.append(p)

        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1

    if also_steep:
        ret.reverse()

    return ret