#!/usr/bin/env python -u
import cv2
import numpy
import os
import pickle
import sys


if len(sys.argv) != 3:
    sys.exit('Usage: {0} pages_directory_path contrast_threshold'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('contrast_threshold', contrast_threshold)

################################################################################

# load the each page of the corpus as grey scale, detect its keypoints and
# compute its descriptors
print 'Loading pages...'
corpus = {'pages': list(), 'keypoints': list(), 'descriptors': list()}

sift = cv2.SIFT(contrastThreshold=contrast_threshold)
for page_file_path in os.listdir(pages_directory_path):
    print '   Detecting keypoints and computing descriptors on \'{0}\'...'.format(page_file_path)
    page = cv2.imread(pages_directory_path + '/' + page_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    page_keypoints, page_descriptors = sift.detectAndCompute(page, None)
    # page_keypoints: the input/output vector of keypoints
    # page_descriptors: the output matrix of descriptors
    page_keypoints = numpy.asarray([keypoint.pt for keypoint in page_keypoints])

    corpus['pages'].append(page_file_path)
    corpus['keypoints'].append(page_keypoints)
    corpus['descriptors'].append(page_descriptors)


# save pages, keypoints and descriptors for later use
print 'Saving corpus...'
with open('corpus-' + str(contrast_threshold), 'wb') as f:
    pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
