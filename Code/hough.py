import cv2
import numpy as np
import sys
from PIL import Image
import math
from scipy import ndimage as ndi

#cv2.imshow('image1',image1)
#cv2.waitKey(0)
#cv2.destoryAllWindows(0)

# read file
img = cv2.imread('../Data/ortho/0153359e_582245n_20160905T073406Z_tex.tif')
cv2.imwrite('size.png',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

im = Image.open('size.png')
width, height = im.size
im.load()
print(width + height)

# Canny
edges = cv2.Canny(gray,100,200,apertureSize = 3) #Min och Max gradient values, higher thresholds; fewer lines detected

cv2.imshow('Canny',edges)
cv2.waitKey(0)

# Hough

minLineLength = 0.001
maxLineGap = 0.1
lines = cv2.HoughLinesP(edges, 1, np.pi/360, 20, minLineLength, maxLineGap)


for x in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[x]:
       cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


# save and show
cv2.imwrite('houghlines3.png',img)
hough = Image.open("houghlines3.png")

original = Image.open("size.png")

if hough.size != original.size:
	print("Different sizes in image; hough size: " + hough.size + "original size : " + original.size)

if hough.mode != original.mode:
	print("Different modes in images; hough mode: " + hough.mode + "original mode: " + original.mode)

c = Image.blend(hough,original,0.3) #lower for less original
c.save("abc.png", "PNG")

hough_img = cv2.imread('abc.png')

Image.fromarray(hough_img).show()







