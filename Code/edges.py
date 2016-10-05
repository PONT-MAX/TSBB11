import numpy as np
import cv2


def get_edges(ortho,object_mask):

    gray_image = cv2.cvtColor(ortho, cv2.COLOR_BGR2GRAY)
    object_mask = np.multiply(gray_image,object_mask)

    object_mask=np.uint8(object_mask)

    edge = cv2.Canny(object_mask,100,200)
    #lines = cv2.HoughLinesP(edge,1,np.pi/180,100,50,5)

    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(edge, 1, np.pi /180, 15, minLineLength, maxLineGap)
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(ortho, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('hough', ortho)
    cv2.waitKey(0)

    return edge