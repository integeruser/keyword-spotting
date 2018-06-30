#!/usr/bin/env python2 -u
import cv2
import numpy
import sys


if len(sys.argv) != 4:
    sys.exit('Usage: {0} image1_file_path n_features n_octave_layers'.format(sys.argv[0]))

image1_file_path = sys.argv[1]
n_features = int(sys.argv[2])
n_octave_layers = int(sys.argv[3])
print 'Starting script...'
print '   {0: <18} = {1}'.format('image1_file_path', image1_file_path)
print '   {0: <18} = {1}'.format('n_features', n_features)
print '   {0: <18} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

def prune_near_keypoints(keypoints, pixel_threshold=1):
    pruned_keypoints = list()
    for i, keypoint1 in enumerate(keypoints):
        to_prune = False
        for keypoint2 in keypoints[i+1:]:
            dist = abs(numpy.linalg.norm(
                numpy.asarray(keypoint1.pt) - numpy.asarray(keypoint2.pt)))
            if dist <= pixel_threshold:
                to_prune = True
                break
        if not to_prune:
            pruned_keypoints.append(keypoint1)

    return pruned_keypoints


sift = cv2.SIFT(nfeatures=n_features, nOctaveLayers=n_octave_layers)

image1 = cv2.imread(image1_file_path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

keypoints1 = sift.detect(image1, None)
pruned_keypoints1 = prune_near_keypoints(keypoints1)
print 'original # keypoints: {n}'.format(n=len(keypoints1))
print 'pruned # keypoints:   {n}'.format(n=len(pruned_keypoints1))

cv2.imshow('image1',
           cv2.drawKeypoints(image1, keypoints1, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.imshow('image1-pruned',
           cv2.drawKeypoints(image1, pruned_keypoints1, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey()
