#!/usr/bin/env python2 -u
import cv2
import itertools
import numpy
import re
import sys
import utils


if len(sys.argv) != 5:
    sys.exit('Usage: {0} corpus_file_path codebook_size max_iter epsilon'.format(sys.argv[0]))

corpus_file_path = sys.argv[1]
codebook_size = int(sys.argv[2])
max_iter = int(sys.argv[3])
epsilon = float(sys.argv[4])
print 'Starting script...'
print '   {0: <16} = {1}'.format('corpus_file_path', corpus_file_path)
print '   {0: <16} = {1}'.format('codebook_size', codebook_size)
print '   {0: <16} = {1}'.format('max_iter', max_iter)
print '   {0: <16} = {1}'.format('epsilon', epsilon)

################################################################################

# load precomputed keypoints and descriptors
print 'Loading corpus...'
corpus = utils.load_corpus(corpus_file_path)
assert len(corpus['pages']) > 0
assert len(corpus['pages']) == len(corpus['keypoints']) == len(corpus['descriptors'])
for page_keypoints, page_descriptors in itertools.izip(corpus['keypoints'], corpus['descriptors']):
    assert len(page_keypoints) == len(page_descriptors)


# run k-means on the descriptors space
print 'Running k-means on the descriptors space...'
corpus_descriptors_vstack = numpy.vstack(corpus['descriptors'])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
attempts = 10
compactness, labels, d = cv2.kmeans(
    corpus_descriptors_vstack, codebook_size, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
assert len(d) == codebook_size


# construct the codebook of k visual words
print 'Constructing codebook...'
corpus_keypoints_vstack = numpy.asarray(
    [keypoint for page_keypoints in corpus['keypoints'] for keypoint in page_keypoints])

codebook = {
    'corpus_file_path': corpus_file_path,
    'codebook_size': codebook_size,
    'max_iter': max_iter,
    'epsilon': epsilon,
    'codewords': list()
}
# create a codeword for each k-means group found
for group in range(len(d)):
    codeword_centroid = d[group]

    # find keypoints that belong to the current group, and group them in pages
    codeword_keypoints = dict.fromkeys(corpus['pages'], list())
    codeword_descriptors = dict.fromkeys(corpus['pages'], list())

    curr_page_start_index = curr_page_last_index = 0
    for i, page in enumerate(corpus['pages']):
        assert len(corpus['keypoints'][i]) == len(corpus['descriptors'][i])
        curr_page_last_index = curr_page_start_index + len(corpus['keypoints'][i])

        curr_page_keypoints = corpus_keypoints_vstack[
            curr_page_start_index:curr_page_last_index]
        curr_page_curr_group_keypoints = curr_page_keypoints[
            labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
        codeword_keypoints[page] = curr_page_curr_group_keypoints

        curr_page_descriptors = corpus_descriptors_vstack[
            curr_page_start_index:curr_page_last_index]
        curr_page_curr_group_descriptors = curr_page_descriptors[
            labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
        codeword_descriptors[page] = curr_page_curr_group_descriptors

        curr_page_start_index = curr_page_last_index

    codeword = {
        'd': codeword_centroid,
        'keypoints': codeword_keypoints,
        'descriptors': codeword_descriptors
    }
    codebook['codewords'].append(codeword)

assert len(codebook['codewords']) == codebook_size
assert len(corpus_keypoints_vstack) == len(corpus_descriptors_vstack)
assert sum([len(v) for codeword in codebook['codewords'] for v in codeword['keypoints'].values()]) == len(
    corpus_keypoints_vstack)
assert sum([len(v) for codeword in codebook['codewords'] for v in codeword['descriptors'].values()]) == len(
    corpus_descriptors_vstack)


# save codebook for later use
print 'Saving codebook...'
contrast_threshold = re.findall(r'corpus-.*-([^(]*)-.*', corpus_file_path)[0]
n_octave_layers = corpus_file_path[corpus_file_path.rfind('-') + 1:]
codebook_file_name = 'codebook-{0}-{1}-{2}--{3}-{4}-{5}'.format(
    len(corpus['pages']), contrast_threshold, n_octave_layers, codebook_size, max_iter, epsilon)
utils.save_codebook(codebook, codebook_file_name)
