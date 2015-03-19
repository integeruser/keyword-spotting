#!/usr/bin/env python -u
import cv2
import numpy
import os
import cPickle
import sys


if len(sys.argv) != 4:
    sys.exit(
        'Usage: {0} pages_directory_path contrast_threshold n_octave_layers'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
n_octave_layers = int(sys.argv[3])
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('contrast_threshold', contrast_threshold)
print '   {0: <20} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

# load the each page of the corpus as grey scale, detect its keypoints and compute its descriptors
print 'Loading pages...'
corpus = {
    'pages': list(),
    'keypoints': list(),
    'descriptors': list()
}

sift = cv2.SIFT(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)

for page_file_path in os.listdir(pages_directory_path):
    print '   Detecting keypoints and computing descriptors on \'{0}\'...'.format(page_file_path)
    page_image = cv2.imread(
        pages_directory_path + '/' + page_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    page_keypoints, page_descriptors = sift.detectAndCompute(page_image, None)

    # we can't serialize directly cv2.KeyPoint, so we create a tuple that resemble it
    page_keypoints_serializable = list()
    for keypoint in page_keypoints:
        keypoint_serializable = (
            keypoint.pt, keypoint.size, keypoint.angle,
            keypoint.response, keypoint.octave, keypoint.class_id)
        page_keypoints_serializable.append(keypoint_serializable)

    corpus['pages'].append(page_file_path)
    corpus['keypoints'].append(page_keypoints_serializable)
    corpus['descriptors'].append(page_descriptors)


# save pages, keypoints and descriptors for later use
print 'Saving corpus...'
corpus_file_name = 'corpus-{0}-{1}-{2}'.format(
    len(corpus['pages']), contrast_threshold, n_octave_layers)
with open(corpus_file_name, 'wb') as f:
    cPickle.dump(corpus, f, protocol=cPickle.HIGHEST_PROTOCOL)
