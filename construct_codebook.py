#!/usr/bin/env python -u
import collections
import cv2
import itertools
import numpy
import sys


Codeword = collections.namedtuple('Codeword', 'd features')
Feature = collections.namedtuple('Feature', 'p x')


if len(sys.argv) != 4:
    sys.exit(
        'Usage: {0} keypoints_file_path descriptors_file_path k'.format(sys.argv[0]))

keypoints_file_path = sys.argv[1]
descriptors_file_path = sys.argv[2]
k = int(sys.argv[3])
print 'Starting script...'
print '   {0: >21} = {1}'.format('keypoints_file_path', keypoints_file_path)
print '   {0: >21} = {1}'.format('descriptors_file_path', descriptors_file_path)
print '   {0: >21} = {1}'.format('k', k)

################################################################################

# load precompute keypoints and descriptors
print 'Loading \'{0}\' and \'{1}\'...'.format(keypoints_file_path,
                                              descriptors_file_path)
corpus_keypoints = numpy.load(keypoints_file_path)
corpus_descriptors = numpy.load(descriptors_file_path)
assert(len(corpus_keypoints) == len(corpus_descriptors))
for page_keypoints, page_descriptors in itertools.izip(corpus_keypoints,
                                                       corpus_descriptors):
    assert(len(page_keypoints) == len(page_descriptors))

# run k-means on the descriptors space
print 'Running k-means (k={0}) on the descriptors space...'.format(k)
corpus_keypoints_vstack = numpy.vstack(corpus_keypoints)
corpus_descriptors_vstack = numpy.vstack(corpus_descriptors)

max_iter = 10
epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            max_iter, epsilon)
attempts = 10
compactness, labels, d = cv2.kmeans(corpus_descriptors_vstack, k,
                                    criteria, attempts,
                                    cv2.KMEANS_RANDOM_CENTERS)
# compactness: the sum of squared distance from each point to their
#              corresponding centers
# labels: the label array where each element marked '0', '1', ...
# d: the array of centers of clusters

# construct the codebook of k visual words
print 'Constructing codebook...'
codebook = list()

for group in range(k):
    centroid = d[group]

    # iterate corpus keypoints per page
    features = list()
    curr_page_start_index = 0
    curr_page_last_index = 0
    for page in range(len(corpus_keypoints)):
        curr_page_last_index = curr_page_start_index + \
            len(corpus_keypoints[page])

        curr_page_keypoints = corpus_keypoints_vstack[
            curr_page_start_index:curr_page_last_index]
        curr_page_curr_group_keypoints = curr_page_keypoints[
            labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
        for keypoint in curr_page_curr_group_keypoints:
            features.append(Feature(page, keypoint))

        curr_page_start_index = curr_page_last_index
    codebook.append(Codeword(centroid, features))
assert(len(codebook) == k)
assert(sum([len(codeword.features)
            for codeword in codebook]) == len(corpus_keypoints_vstack))

# save codebook for later use
print 'Saving codebook...'
numpy.save('codebook', codebook)
