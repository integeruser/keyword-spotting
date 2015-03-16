#!/usr/bin/env python -u
import cv2
import itertools
import numpy
import pickle
import sys


if len(sys.argv) != 3:
    sys.exit('Usage: {0} query_file_path codebook_file_path'.format(sys.argv[0]))

query_file_path = sys.argv[1]
codebook_file_path = sys.argv[2]
print 'Starting script...'
print '   {0: <18} = {1}'.format('query_file_path', query_file_path)
print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)

################################################################################

# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
query = cv2.imread(query_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

sift = cv2.SIFT()
query_keypoints, query_descriptors = sift.detectAndCompute(query, None)
# query_keypoints: the input/output vector of keypoints
# query_descriptors: the output matrix of descriptors
query_keypoints = numpy.asarray([keypoint.pt for keypoint in query_keypoints])


# find the nearest codewords to the query descriptors
print 'Loading the codebook...'
with open(codebook_file_path, 'rb') as f:
    codebook = pickle.load(f)

q = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook]
    nearest_codeword_index = numpy.argmin(query_descriptor_distances)

    q.append({'w': nearest_codeword_index, 'x': query_keypoints[i]})
