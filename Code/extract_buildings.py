import numpy as np
import cv2
import math
from PIL import Image
import scipy
from scipy import ndimage
import help_functions


def get_buildings(ortho,object_mask):

    ortho_clean=ortho.copy()

    gray_image = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    object_mask = np.multiply(gray_image,object_mask)

    #grayscale image where the unwanted parts=0
    object_mask=np.uint8(object_mask)

    #detecting/counting ht enumber of blobs in the image
    number_of_blobs, labels, stats, ortho_png = help_functions.detecting_blobs(object_mask, ortho)

    #labeling each blob to be able to use the for-loop on them
    labels=np.uint8(labels)
    _, thresh1 = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)
    thresh1_clean=thresh1.copy()

    #getting coords, size, etc on each blob aswell as the total number of blobs
    number_of_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=4)

    #making a 3 channel image of the thresholded image
    thresh_3d = np.repeat(thresh1[:, :, np.newaxis], 3, axis=2)
    thresh_3dclean = thresh_3d.copy()

    #extracting and drawing the bounding box of each object
    for blob_no in range(1, number_of_blobs):
        left_coord=stats[blob_no, 0]
        top_coord=stats[blob_no,1]
        width=stats[blob_no,2]
        height=stats[blob_no,3]
        cv2.rectangle(thresh_3d, (left_coord, top_coord), (left_coord + width, top_coord + height), (0, 255, 0), 2)

    patch_list = []
    coords_list = []

    #start at 1 since first blob is the background
    for box_no in range(40, 50):
        im_patch, im_coords = help_functions.crop_im_part(thresh1_clean, stats[box_no, :])
        patch_list.append(im_patch)
        coords_list.append(im_coords)
        #rotated_boxes = help_functions.get_rotated_box(im_patch, 2)
        cv2.imshow('hough', im_patch)
        cv2.waitKey(0)
    """

   for box_no in range(40, 50):
       im_patch = help_functions.crop_im_part(thresh_3dclean, stats[box_no, :])
       h = help_functions.hough_on_part(im_patch)
       cv2.imshow('hough', h)
       cv2.waitKey(0)
   """

    #rotated_boxes=help_functions.get_rotated_box(labels, number_of_blobs)
    #Image.fromarray(rotated_boxes).show()
    return thresh_3d, patch_list, coords_list

    #see which components have a number of pixels that are above a certain number of connected pixels and open/close them
    #but not the smaller ones...



#merged buildings, watershed-ish?
#villas smaller than noise
#bounding boxes horizontal instead of angled
#bounding boxes only rectangular, what if weirdly shaped house
#extracted blob regions that contains whole or parts of other blobs
