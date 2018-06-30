#!/usr/bin/env python2 -u
import cv2
import itertools
import numpy
import sys
import utils


if len(sys.argv) != 5:
    sys.exit('Usage: {0} codebook_file_path query_file_path n_features rho'.format(sys.argv[0]))

codebook_file_path = sys.argv[1]
query_file_path = sys.argv[2]
n_features = int(sys.argv[3])
rho = float(sys.argv[4])
print 'Starting script...'
print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <18} = {1}'.format('query_file_path', query_file_path)
print '   {0: <18} = {1}'.format('n_features', n_features)
print '   {0: <18} = {1}'.format('rho', rho)

################################################################################

def prune_near_keypoints(keypoints, descriptors, pixel_threshold=1):
    pruned_keypoints = list()
    pruned_descriptors = list()
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
            pruned_descriptors.append(descriptors[i])

    return pruned_keypoints, pruned_descriptors



# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
sift = cv2.SIFT(nfeatures=n_features, nOctaveLayers=1)

query_image = cv2.imread(query_file_path)
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
assert len(query_keypoints) > 0
assert len(query_keypoints) == len(query_descriptors)
print '   Keypoints detected: {0}'.format(len(query_keypoints))

# prune near keypoints
query_keypoints, query_descriptors = prune_near_keypoints(query_keypoints, query_descriptors)
print '   Keypoints after pruning: {0}'.format(len(query_keypoints))
print '   A match must contain at least {n} keypoints (since rho={rho})'.format(n=int(len(query_keypoints)*rho),
                                                                                rho=rho)


# load the codebook
print 'Loading the codebook...'
codebook = utils.load_codebook(codebook_file_path)
print '   Codebook keypoints count: {0}'.format(sum([len(v) for codeword in codebook['codewords']
                                                     for v in codeword['keypoints'].viewvalues()]))

# find the nearest codewords to the query descriptors
print 'Finding the nearest codewords to the query...'
query_points = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook['codewords']]
    nearest_codeword = codebook['codewords'][numpy.argmin(query_descriptor_distances)]

    query_point = {
        'w': nearest_codeword,
        'x': query_keypoints[i]
    }
    query_points.append(query_point)


# find matches in each page
print 'Finding matches in each page...'
matches = dict()
for page in codebook['pages']:
    keypoints_count = sum([len(query_point['w']['keypoints'][page]) for query_point in query_points])
    err_matrix = numpy.zeros((keypoints_count, keypoints_count))

    row_index = 0
    for i, query_point1 in enumerate(query_points):
        codeword1_keypoints = query_point1['w']['keypoints'][page]
        query_keypoint1 = query_point1['x']
        for keypoint1 in codeword1_keypoints:
            col_index = 0
            for query_point2 in query_points[:i]:
                codeword2_keypoints = query_point2['w']['keypoints'][page]
                query_keypoint2 = query_point2['x']
                for keypoint2 in codeword2_keypoints:
                    # compute distances between two query points and two page keypoints
                    query_keypoints_diff = (query_keypoint1.pt[0] - query_keypoint2.pt[0])**2 + \
                                           (query_keypoint1.pt[1] - query_keypoint2.pt[1])**2
                    keypoints_diff       = (keypoint1.pt[0] - keypoint2.pt[0])**2 + \
                                           (keypoint1.pt[1] - keypoint2.pt[1])**2
                    # the match error is the difference bewteen the distances
                    err_matrix[row_index, col_index] = abs(query_keypoints_diff - keypoints_diff)

                    col_index += 1
            row_index += 1

    assert not numpy.count_nonzero(numpy.triu(err_matrix))  # upper triangular matrix should be 0
    i = 0
    for query_point in query_points:
        l = len(query_point['w']['keypoints'][page])
        assert not numpy.count_nonzero(err_matrix[i:i+l, i:i+l])


    # numpy.set_printoptions(threshold=numpy.nan)
    # f = open('kappa.txt', 'w')
    # for row in err_matrix:
    #     f.write(numpy.array_str(row))
    #     f.write('\n')
    # f.close()
    exit()
