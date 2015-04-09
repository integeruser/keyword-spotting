#!/usr/bin/env python2 -u
import cv2
import numpy
import sys

from matplotlib import pyplot


if len(sys.argv) != 5:
    sys.exit('Usage: {0} image_file_path query_file_path contrast_threshold n_octave_layers'.format(sys.argv[0]))

image_file_path = sys.argv[1]
query_file_path = sys.argv[2]
contrast_threshold = float(sys.argv[3])
n_octave_layers = int(sys.argv[4])
print 'Starting script...'
print '   {0: <18} = {1}'.format('image_file_path', image_file_path)
print '   {0: <18} = {1}'.format('query_file_path', query_file_path)
print '   {0: <18} = {1}'.format('contrast_threshold', contrast_threshold)
print '   {0: <18} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold,
                                   nOctaveLayers=n_octave_layers)

img1 = cv2.imread(query_file_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(image_file_path)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
pyplot.imshow(img3)
pyplot.show()
