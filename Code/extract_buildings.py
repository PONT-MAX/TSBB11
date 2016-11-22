import numpy as np
import cv2
import math
from PIL import Image
import scipy
from scipy import ndimage
import help_functions


def get_buildings(ortho,object_mask, dhm):
    #THE DHM IS NORMALIZED!!!!

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
    #thresh1_clean=thresh1.copy()

    #Image.fromarray(thresh1).show()

    #getting coords, size, etc on each blob aswell as the total number of blobs
    number_of_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(thresh1, connectivity=4)

    #find the largest blobs and perforn more morphology on them
    meanval=np.mean(stats, axis=0)[4]
    better_morph=help_functions.more_morph(thresh1, labels, number_of_blobs, stats, meanval, dhm)

    #more morphology on the blobs from the largest blobs
    kernel = np.ones((7,7),np.uint8)
    better_morph = cv2.morphologyEx(better_morph, cv2.MORPH_CLOSE, kernel)
    better_morph = cv2.morphologyEx(better_morph, cv2.MORPH_OPEN, kernel)
    Image.fromarray(better_morph).show()

    #making a 3 channel image of the thresholded image
    #thresh_3d = np.repeat(thresh1[:, :, np.newaxis], 3, axis=2)
    #thresh_3dclean = thresh_3d.copy()

    """
    patch_list = []
    coords_list = []

    #start at 1 since first blob is the background
    margin=20
    for box_no in range(40, 50):
        im_patch, im_coords = help_functions.crop_im_part(thresh1_clean, stats[box_no, :],margin)
        patch_list.append(im_patch)
        coords_list.append(im_coords)

        #no_blobs, patch_labels, _, _ = cv2.connectedComponentsWithStats(im_patch, connectivity=4)
        #rotated_boxes = help_functions.get_rotated_box(patch_labels , no_blobs)

        #cv2.imshow('hough', rotated_boxes)
        #cv2.waitKey(0)
    """

    rotated_box = help_functions.get_rotated_box(better_morph)

    return rotated_box
    #return thresh_3d

    #see which components have a number of pixels that are above a certain number of connected pixels and open/close them
    #but not the smaller ones...
    #or significantly larger than meanvalue, a certain percentage?



#merged buildings, watershed-ish?
#villas smaller than noise
#bounding boxes horizontal instead of angled
#bounding boxes only rectangular, what if weirdly shaped house
#extracted blob regions that contains whole or parts of other blobs
