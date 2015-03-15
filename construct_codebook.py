#!/usr/bin/env python -u
import cv2
import itertools
import numpy
import pickle
import sys


if len(sys.argv) != 3:
    sys.exit('Usage: {0} corpus_file_path k'.format(sys.argv[0]))

corpus_file_path = sys.argv[1]
k = int(sys.argv[2])
print 'Starting script...'
print '   {0: <16} = {1}'.format('corpus_file_path', corpus_file_path)
print '   {0: <16} = {1}'.format('k', k)

################################################################################

# load precomputed keypoints and descriptors
print 'Loading corpus...'
with open(corpus_file_path, 'rb') as f:
    corpus = pickle.load(f)

assert(len(corpus['pages']) == len(corpus['keypoints']) == len(corpus['descriptors']))
for page_keypoints, page_descriptors in itertools.izip(corpus['keypoints'], corpus['descriptors']):
    assert(len(page_keypoints) == len(page_descriptors))


# run k-means on the descriptors space
print 'Running k-means on the descriptors space...'
corpus_keypoints_vstack = numpy.vstack(corpus['keypoints'])
corpus_descriptors_vstack = numpy.vstack(corpus['descriptors'])

max_iter = 10
epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
attempts = 10
compactness, labels, d = cv2.kmeans(
    corpus_descriptors_vstack, k, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
# compactness: the sum of squared distance from each point to their corresponding centers
# labels: the label array where each element marked '0', '1', ...
# d: the array of centers of clusters


# construct the codebook of k visual words
print 'Constructing codebook...'
codebook = list()

# create a codeword for each k-means group found
for group in range(k):
    centroid = d[group]

    # find which keypoints belong to the current group
    features = list()
    curr_page_start_index = 0
    curr_page_last_index = 0
    for page in range(len(corpus['pages'])):
        curr_page_last_index = curr_page_start_index + \
            len(corpus['keypoints'][page])

        curr_page_keypoints = corpus_keypoints_vstack[
            curr_page_start_index:curr_page_last_index]
        curr_page_curr_group_keypoints = curr_page_keypoints[
            labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
        for keypoint in curr_page_curr_group_keypoints:
            feature = {'p': page, 'x': keypoint}
            features.append(feature)

        curr_page_start_index = curr_page_last_index

    codeword = {'d': centroid, 'features': features}
    codebook.append(codeword)

assert(len(codebook) == k)
assert(sum([len(codeword['features']) for codeword in codebook]) == len(corpus_keypoints_vstack))


# save codebook for later use
print 'Saving codebook...'
with open('codebook-' + str(k), 'wb') as f:
    pickle.dump(codebook, f, protocol=pickle.HIGHEST_PROTOCOL)
