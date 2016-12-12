import numpy as np
import cv2
import math
from PIL import Image
import scipy
from scipy import ndimage
import help_functions

def getBuildings(ortho,object_mask, dhm, minBlob, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES):
    #Note: dhm is normalized
    ortho_clean=ortho.copy()

    gray_image = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    object_mask = np.multiply(gray_image,object_mask)

    #grayscale image where the unwanted parts=0
    object_mask=np.uint8(object_mask)

    #detecting/counting ht enumber of blobs in the image

    number_of_blobs, labels, stats, ortho_png = help_functions.getBlobs(object_mask, ortho, minBlob)

    #labeling each blob to be able to use the for-loop on them
    labels=np.uint8(labels)
    _, thresh1 = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)

    #thresh1_clean=thresh1.copy()

    #Image.fromarray(thresh1).show()

    #getting coords, size, etc on each blob aswell as the total number of blobs
    number_of_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(thresh1, connectivity=4)

    #find the largest blobs and perforn more morphology on them
    meanval=np.mean(stats, axis=0)[4]
    better_morph=help_functions.moreMorph(thresh1, labels, number_of_blobs, stats, meanval, dhm)

    #more morphology on the blobs from the largest blobs
    kernel = np.ones((7,7),np.uint8)
    better_morph = cv2.morphologyEx(better_morph, cv2.MORPH_CLOSE, kernel)
    better_morph = cv2.morphologyEx(better_morph, cv2.MORPH_OPEN, kernel)

    approxBoxes = help_functions.getApproxBoxes(better_morph, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES)

    return approxBoxes, better_morph

    #see which components have a number of pixels that are above a certain number of connected pixels and open/close them
    #but not the smaller ones...
    #or significantly larger than meanvalue, a certain percentage?
