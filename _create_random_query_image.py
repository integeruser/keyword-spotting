#!/usr/bin/env python -u
import cv2
import os
import random
import sys


if len(sys.argv) != 3:
    sys.exit('Usage: {0} pages_directory_path boxes_directory_path'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
boxes_directory_path = sys.argv[2]
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('boxes_directory_path', boxes_directory_path)

################################################################################

# load a random boxes file
print 'Loading boxes...'
boxes_file = random.choice(os.listdir(boxes_directory_path))
print '   Boxes file chosen: {0}'.format(boxes_file)

with open(boxes_directory_path + '/' + boxes_file) as f:
    boxes_file_content = f.readlines()


# retrieve the page file name from the boxes file
boxes_file_name = boxes_file_content[0]
page_file_name = boxes_file_name[:boxes_file_name.index('_')] + '.tif'
print '   Page file chosen: {0}'.format(page_file_name)


# chose a random box
print 'Loading a random box...'
random_box = random.choice(boxes_file_content[1:])
points = random_box.split()
print '   {0}'.format(points)


# extract the box from its page
page = cv2.imread(pages_directory_path + '/' + page_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
page_height, page_width = page.shape[:2]
x1 = int(float(points[0]) * page_width)
x2 = int(float(points[1]) * page_width)
y1 = int(float(points[2]) * page_height)
y2 = int(float(points[3]) * page_height)
pt1 = (x1, y1)
pt2 = (x2, y2)
print '   Pt1: {0}'.format(pt1)
print '   Pt2: {0}'.format(pt2)

query_file_name = 'query-' + str(pt1) + '-' + str(pt2) + '.png'
cv2.imwrite(query_file_name, page[y1:y2, x1:x2])
