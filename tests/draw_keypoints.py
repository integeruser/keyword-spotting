#!/usr/bin/env python2 -u
import cv2
import sys


if len(sys.argv) != 4:
    sys.exit('Usage: {0} image_file_path contrast_threshold n_octave_layers'.format(sys.argv[0]))

image_file_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
n_octave_layers = int(sys.argv[3])
print 'Starting script...'
print '   {0: <18} = {1}'.format('image_file_path', image_file_path)
print '   {0: <18} = {1}'.format('contrast_threshold', contrast_threshold)
print '   {0: <18} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold,
                                   nOctaveLayers=n_octave_layers)

image = cv2.imread(image_file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

keypoints = sift.detect(image, None)

cv2.imshow('keypoints-contrast_threshold',
           cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey()
