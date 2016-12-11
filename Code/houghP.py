import cv2
import numpy as np
from help_functions import lineIter

'''
Probabilistic Hough transform applied on a smaller region around a building
	Input parameters
	img:			satellite image as numpy array 
	patch:			binary mask patch for a house region as numpy array
	coord:			list with coordinates for patch as 
					[x_start, x_end, y_start, y_end]
	all_channels:	bool which is set to True if additional operation 
				 	should be done on every color channel separatly,
				 	if False it is only done on gray image

	Output
	part_of_image:	corresponding patch in satellite image with lines
					retrieved from Hough transform drawn
	selected_lines:	list of lines retrieved as x_start, y_start, x_end, y_end
					for every line
'''

def houghP(img, patch, coord, all_channels):

	part_of_image = img[coord[0]:coord[1], coord[2]:coord[3]]
	
	gray_image = cv2.cvtColor(part_of_image, cv2.COLOR_BGR2GRAY)
	b,g,r = cv2.split(part_of_image)

	# Canny

	sigma = 0.4
	v_gray = np.median(part_of_image)
	lower_gray = int(max(0, (1.0 - sigma) * v_gray))
	upper_gray = int(min(255, (1.0 + sigma) * v_gray))
	
	if all_channels:
		sigma = 0.35 #0.1 or 0.3
		v_blue = np.median(b)
		lower_blue = int(max(0, (1.0 - sigma) * v_blue))
		upper_blue = int(min(255, (1.0 + sigma) * v_blue))

		sigma = 0.35
		v_green = np.median(g)
		lower_green = int(max(0, (1.0 - sigma) * v_green))
		upper_green = int(min(255, (1.0 + sigma) * v_green))

		sigma = 0.35
		v_red = np.median(r)
		lower_red = int(max(0, (1.0 - sigma) * v_red))
		upper_red = int(min(255, (1.0 + sigma) * v_red))

		edges_red = cv2.Canny(r,lower_red,upper_red,apertureSize = 3)
		edges_green = cv2.Canny(g,lower_green,upper_green,apertureSize = 3)
		edges_blue = cv2.Canny(b,lower_blue,upper_blue,apertureSize = 3)
	
	edges = cv2.Canny(part_of_image, lower_gray, upper_gray,apertureSize = 3, L2gradient=True)

    # Hough
	threshold = 10 
	minLineLength = 10
	maxLineGap = 0.1
	total = cv2.HoughLinesP(edges, 1, np.pi/360, threshold, minLineLength, maxLineGap)

	if total is None:
		print("No lines found in image patch")
		return img, total
	else:
		print("number of gray lines found: " + str(total.shape[0]))
		if all_channels:
			lines_red = cv2.HoughLinesP(edges_red, 1, np.pi/360, threshold, minLineLength, maxLineGap)
			lines_green = cv2.HoughLinesP(edges_green, 1, np.pi/360, threshold, minLineLength, maxLineGap)
			lines_blue = cv2.HoughLinesP(edges_blue, 1, np.pi/360, threshold, minLineLength, maxLineGap)

			all_lines = [lines_red, lines_green, lines_blue]
			
			if lines_red is not None:
				print("number of red lines found: " + str(lines_red.shape[0]))
			if lines_green is not None:
				print("number of green lines found: " + str(lines_green.shape[0]))
			if lines_blue is not None:
				print("number of blue lines found: " + str(lines_blue.shape[0]))
			
		
			for lines in all_lines:
				if lines is not None:
					total = np.concatenate((total, lines))

		# Select lines intersecting with blob

		part_of_image_clean = img[coord[0]:coord[1], coord[2]:coord[3]]
		
		selected_lines, select_lines_itered, blob_coords= selectLines(patch, total)
		for x in range(0, len(selected_lines)):
			for x1, y1, x2, y2 in selected_lines[x]:
			   cv2.line(part_of_image_clean, (x1, y1), (x2, y2), (0, 255, 0), 2)

		cv2.imshow('hough', part_of_image)
		cv2.waitKey(0)
		
		return part_of_image, selected_lines

# Remove all hough lines which do not intersect with blob from binary mask
def selectLines(patch, line_list):

	# get pixelcoordinates where blob is white
	y,x = np.where(patch == 255)
	blob_coords = list(zip(x,y))

	intersected_lines = []
	intersected_lines_itered = []
	for i in range(0, len(line_list)):
		itered_line = lineIter(patch, line_list[i])
		intersected_coords = set(itered_line).intersection(blob_coords)
		if intersected_coords:
			intersected_lines.append(line_list[i])
			intersected_lines_itered.append(itered_line)
	return intersected_lines, intersected_lines_itered, blob_coords


