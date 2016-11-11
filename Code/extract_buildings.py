import numpy as np
import cv2
import math
from PIL import Image
import scipy
from scipy import ndimage
import help_functions


def getBuildings(ortho,object_mask):
    ortho_clean=ortho.copy()

    gray_image = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    object_mask = np.multiply(gray_image,object_mask)

    #grayscale image where the unwanted parts=0
    object_mask=np.uint8(object_mask)

    #detecting/counting ht enumber of blobs in the image
    number_of_blobs, labels, stats, ortho_png = help_functions.getBlobs(object_mask, ortho)

    #labeling each blob to be able to use the for-loop on them
    labels=np.uint8(labels)
    _, thresh1 = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)

    return thresh1

    #see which components have a number of pixels that are above a certain number of connected pixels and open/close them
    #but not the smaller ones...



#merged buildings, watershed-ish?
#villas smaller than noise
#bounding boxes horizontal instead of angled
#bounding boxes only rectangular, what if weirdly shaped house
#extracted blob regions that contains whole or parts of other blobs
