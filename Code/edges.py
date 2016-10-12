import numpy as np
import cv2
import math
from PIL import Image


def get_edges(ortho,object_mask):

    gray_image = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    object_mask = np.multiply(gray_image,object_mask)

    object_mask=np.uint8(object_mask)

   # median = cv2.medianBlur(object_mask, 5)

    #edge = cv2.Canny(median,100,200)




    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(object_mask, 1, np.pi/180, 30, minLineLength, maxLineGap)


    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
           cv2.line(ortho, (x1, y1), (x2, y2), (0, 255, 0), 2)

    """
    a, b, c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(ortho, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)"""


    cv2.imwrite("ortho_png.png", ortho)
    ortho_png=Image.open("ortho_png.png")
    width, height = ortho_png.size

    img=np.zeros((width,height), np.uint8)
    c = (0, 255, 0)
    indices = np.where(np.all(ortho == c, axis=-1))
    #coords= zip(indices[0], indices[1])
    # dont run this lineeeeeeeee! print(coords)
    img[indices]=255

    kernel = np.ones((7,7),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    median = cv2.medianBlur(img, 9)
    closed = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((10,10),np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)



    return opened