import numpy as np
import cv2
import math
from PIL import Image
import scipy

"""
def watershed(binary_im):
    #binary_im = np.uint8(binary_im)
    #_, binary_im = cv2.threshold(binary_im, 0, 255, cv2.THRESH_BINARY)
    binary_im = cv2.cvtColor(binary_im, cv2.COLOR_BGR2GRAY)
    print binary_im.dtype
    kernel = np.ones((3, 3), np.uint8)

    # Finding sure background area
    sure_bg = cv2.dilate(binary_im, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(binary_im, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1


    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    binary_im = np.repeat(binary_im[:, :, np.newaxis], 3, axis=2)
    markers = cv2.watershed(binary_im, markers)
    binary_im[markers == -1] = [255, 0, 0]

    return binary_im
"""

def watershed(thresh, img):

    fg = cv2.erode(thresh, None, iterations=2)
    bgt = cv2.dilate(thresh, None, iterations=3)
    ret, bg = cv2.threshold(bgt, 1, 128, 1)

    marker = cv2.add(fg, bg)
    marker32 = np.int32(marker)
    cv2.watershed(img, marker32)
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img, img, mask=thresh)

    return res

def detecting_blobs(object_mask, ortho ):

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



def crop_im_part(ortho, stats):
    (h,w) = ortho.shape[:2]

    left_coord = stats[0]
    top_coord = stats[1]
    right_coord = stats[2] + left_coord
    bottom_coord = stats[3] + top_coord
    margin=20

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

    return cropped_im



def hough_on_part(im_part):

    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(im_part, 1, np.pi/180, 30, minLineLength, maxLineGap)

    return lines


def get_rotated_box(bin_im, no_blobs):
    for blob in range(1, no_blobs):
        coords = np.argwhere(bin_im == blob).tolist()
        coords=np.asarray(coords)
        #print coords
        #print type(coords)
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bin_im = cv2.drawContours(bin_im, [box], 0, (255, 255, 0), 5)

    return bin_im

