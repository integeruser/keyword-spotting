#!/usr/bin/env python -u
import cv2
import numpy
import os
import sys


if len(sys.argv) != 3:
    sys.exit(
        'Usage: {0} corpus_directory_path contrast_threshold'.format(sys.argv[0]))

corpus_directory_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
print 'Starting script...'
print '   {0: >21} = {1}'.format('corpus_directory_path', corpus_directory_path)
print '   {0: >21} = {1}'.format('contrast_threshold', contrast_threshold)

################################################################################

# load the each page of the corpus as grey scale, detect its keypoints and
# compute its descriptors
print 'Loading corpus from ' + corpus_directory_path + '...'

sift = cv2.SIFT(contrastThreshold=contrast_threshold)
corpus_keypoints = list()
corpus_descriptors = list()
corpus_pages = list()

page_count = 0
for page_file_path in os.listdir(corpus_directory_path):
    print '   Computing SIFT on \'{0}\'...'.format(page_file_path)
    page = cv2.imread(corpus_directory_path + '/' + page_file_path,
                      cv2.CV_LOAD_IMAGE_GRAYSCALE)

    page_keypoints, page_descriptors = sift.detectAndCompute(page, None)
    # page_keypoints: the input/output vector of keypoints
    # page_descriptors: the output matrix of descriptors

    page_keypoints = numpy.asarray([keypoint.pt for keypoint in page_keypoints])

    corpus_keypoints.append(page_keypoints)
    corpus_descriptors.append(page_descriptors)
    corpus_pages.append((page_file_path, page_count))
    page_count += 1

# save keypoints and descriptors for later use
print 'Saving SIFT data...'
numpy.save('keypoints', corpus_keypoints)
numpy.save('descriptors', corpus_descriptors)
with open('pages', 'w') as f:
    for page in corpus_pages:
        f.write('{0} {1}\n'.format(page[0], page[1]))
