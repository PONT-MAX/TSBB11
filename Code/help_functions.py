import numpy as np
import cv2
import math
from PIL import Image
import scipy
import object

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

    if bottom_coord + 20 > h:
        bottom_coord = h
    else:
        bottom_coord = bottom_coord + margin

    cropped_im = ortho[top_coord:bottom_coord, left_coord:right_coord]

    return cropped_im, [left_coord, right_coord, top_coord, bottom_coord]


def getRotatedBox(bin_im):
    #start at 1 since 0 is background
    #for blob in range(1, no_blobs):
    bin_im_out=bin_im.copy()
    ret, contours, hierarchy = cv2.findContours(bin_im, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE )
    length=len(contours)-1
    feat =np.zeros((length,3))
    for cnt in range(0, length):
        rect = cv2.minAreaRect(contours[cnt+1])
        #w,h,angle in feat_mtx
        #feat[cnt,0]=int(round(rect[1][0]))
        #feat[cnt,1]=int(round(rect[1][1]))
        #feat[cnt,2]=int(round(rect[2]))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(bin_im_out, [box], 0, (255, 0, 0), 2)

        """
        angle = rect[2];
        rect_size = rect[1];
        if (rect[2] < -45.0):
            angle = angle + 90.0
            rect_size[0], rect_size[1] = rect_size[1], rect_size[0]
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        rotated = cv2.warpAffine(bin_im,M,bin_im.shape)
        rotated_patch= cv2.getRectSubPix(rotated, rect_size, rect[0])
        cv2.imshow('hough', rotated_patch)
        cv2.waitKey(0)

        """
    return bin_im_out   #, feat

def getApproxBoxes(bin_im):
    bin_im_out = bin_im.copy()
    ret, contours, hierarchy = cv2.findContours(bin_im,mode = cv2.RETR_EXTERNAL,method =cv2.CHAIN_APPROX_SIMPLE)
    bin_im_out = cv2.merge((bin_im_out,bin_im_out,bin_im_out)) # Making bin_im_out 3 channel to display colors.
    for cnt in contours:
            (x,y), (w,h), angle = cv2.minAreaRect(cnt)
            rectarea = w*h
            area = cv2.contourArea(cnt)
            fillArea = area/rectarea
            if fillArea > 0.65 and area <8000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(bin_im_out,[box],0,(255,0,0),-1)
            # Draw bigger objects with help from perimeter lenght
            else:
                epsilon = 0.01*cv2.arcLength(cnt,True)
                area_cont = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                area_approx = cv2.contourArea(approx)
                fillArea2 = area_approx/area_cont
                cv2.drawContours(bin_im_out,[approx],0,(0,0,255),-1)
                if fillArea2<0.95 and area>10000:
                    epsilon = 0.005*cv2.arcLength(cnt,True)
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

#def getImprovedHouses(a, b, c, d)
##    img_dsm = cv2.imread('dsm_n.png')
#   nice_map = getApproxBoxes(bin_im)



def blobSep(im, dhm):
    #separating buildings in the large blobs into smaller blobs
    im=im/255
    object_mask = np.multiply(dhm,im)
    limit=np.mean(dhm)
    object_mask=object_mask>limit
    object_mask=object_mask*255
    object_mask=np.uint8(object_mask)
    return object_mask


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
        bin_im[stats[large, 1]:(stats[large, 1]+stats[large, 3]), stats[large, 0]:(stats[large, 0]+stats[large, 2])]=separated

    return bin_im

# returns all points in a line
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


