#!/usr/bin/env python2 -u
import cv2
import sys


if len(sys.argv) != 4:
    sys.exit('Usage: {0} image1_file_path contrast_threshold n_octave_layers'.format(sys.argv[0]))

image1_file_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
n_octave_layers = int(sys.argv[3])
print 'Starting script...'
print '   {0: <18} = {1}'.format('image1_file_path', image1_file_path)
print '   {0: <18} = {1}'.format('contrast_threshold', contrast_threshold)
print '   {0: <18} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold,
                                   nOctaveLayers=n_octave_layers)

image1 = cv2.imread(image1_file_path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

keypoints1 = sift.detect(image1, None)

cv2.imshow('image1',
           cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey()
