import cv2
import numpy as np
import sys
from PIL import Image
import math
from scipy import ndimage as ndi
from extract_buildings import get_buildings

#cv2.imshow('image1',image1)
#cv2.waitKey(0)
#cv2.destoryAllWindows(0)

# read file
image = cv2.imread('../Data/ortho/0153359e_582245n_20160905T073406Z_tex.tif')
#img = cv2.imread('../Data/ortho/ej.png')
#cv2.imwrite('size.png',img)

#im = Image.open('size.png')
#width, height = im.size

def hough(img):
	#part_of_image = img[400:750,400:750]
	part_of_image = image
	gray_image = cv2.cvtColor(part_of_image, cv2.COLOR_BGR2GRAY)
	b,g,r = cv2.split(part_of_image)
	# Canny

	sigma = 0.3
	v_gray = np.median(part_of_image)
	lower_gray = int(max(0, (1.0 - sigma) * v_gray))
	upper_gray = int(min(255, (1.0 + sigma) * v_gray))

	sigma = 0.3
	v_blue = np.median(b)
	lower_blue = int(max(0, (1.0 - sigma) * v_blue))
	upper_blue = int(min(255, (1.0 + sigma) * v_blue))

	sigma = 0.3
	v_green = np.median(g)
	lower_green = int(max(0, (1.0 - sigma) * v_green))
	upper_green = int(min(255, (1.0 + sigma) * v_green))

	sigma = 0.3
	v_red = np.median(r)
	lower_red = int(max(0, (1.0 - sigma) * v_red))
	upper_red = int(min(255, (1.0 + sigma) * v_red))


	edges = cv2.Canny(part_of_image, lower_gray, upper_gray,apertureSize = 3, L2gradient=True) #Min och Max gradient values, higher thresholds; fewer lines detected

	edges_red = cv2.Canny(r,lower_red,upper_red,apertureSize = 3)
	edges_green = cv2.Canny(g,lower_green,upper_green,apertureSize = 3)
	edges_blue = cv2.Canny(b,lower_blue,upper_blue,apertureSize = 3)

	# Hough

	minLineLength = 0.001
	maxLineGap = 0.1
	total = cv2.HoughLinesP(edges, 1, np.pi/360, 20, minLineLength, maxLineGap)

	# For every channel
	lines_red = cv2.HoughLinesP(edges_red, 1, np.pi/360, 20, minLineLength, maxLineGap)
	lines_green = cv2.HoughLinesP(edges_green, 1, np.pi/360, 20, minLineLength, maxLineGap)
	lines_blue = cv2.HoughLinesP(edges_blue, 1, np.pi/360, 20, minLineLength, maxLineGap)

	all_lines = [lines_red, lines_green, lines_blue]

	if total is not None:
		print("number of gray lines: " + str(total.shape[0]))
	if lines_red is not None:
		print("number of red lines: " + str(lines_red.shape[0]))
	if lines_green is not None:
		print("number of green lines: " + str(lines_green.shape[0]))
	if lines_blue is not None:
		print("number of blue lines: " + str(lines_blue.shape[0]))

	for lines in all_lines:
		if lines is not None:
			total = np.concatenate((total, lines))

	for x in range(0, len(total)):
		for x1, y1, x2, y2 in total[x]:
		   cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

	return img


img=hough(image)
Image.fromarray(img).show()

# save and show
#cv2.imwrite('houghlines3.png',img)
#hough = Image.open("houghlines3.png")

#original = Image.open("size.png")

#if hough.size != original.size:
#	print("Different sizes in image; hough size: " + hough.size + "original size : " + original.size)

#if hough.mode != original.mode:
#	print("Different modes in images; hough mode: " + hough.mode + "original mode: " + original.mode)

#c = Image.blend(hough,original,0.3) #lower for less original
#c.save("abc.png", "PNG")

#hough_img = cv2.imread('abc.png')





