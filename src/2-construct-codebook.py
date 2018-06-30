#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys

import numpy

import cv2
import utils


def run():
    codebook_fingerprint = os.path.basename(args.corpus_filepath)
    codebook_fingerprint += str(args.codebook_size)
    codebook_fingerprint += str(args.max_iter)
    codebook_fingerprint += str(args.epsilon)
    codebook_fingerprint = codebook_fingerprint.encode('ascii')
    codebook_filepath = f'codebook-{hashlib.sha256(codebook_fingerprint).hexdigest()[:7]}.pickle'
    if os.path.isfile(codebook_filepath):
        msg = 'The codebook already exist. Overwrite?'
        overwrite = input('{} (y/N) '.format(msg)).lower() in ('y', 'yes')
        if not overwrite: sys.exit(1)

    corpus = utils.load_corpus(args.corpus_filepath)
    assert len(corpus['pages']) > 0
    assert len(corpus['pages']) == len(corpus['keypoints']) == len(corpus['descriptors'])
    for page_keypoints, page_descriptors in zip(corpus['keypoints'], corpus['descriptors']):
        assert len(page_keypoints) == len(page_descriptors)

    # run k-means on the descriptors space
    print('Clustering the descriptors space...')
    corpus_descriptors_vstack = numpy.vstack(corpus['descriptors'])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.max_iter, args.epsilon)
    compactness, labels, d = cv2.kmeans(
        corpus_descriptors_vstack, args.codebook_size, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    assert len(d) == args.codebook_size

    corpus_keypoints_vstack = numpy.asarray(
        [keypoint for page_keypoints in corpus['keypoints'] for keypoint in page_keypoints])

    # construct the codebook of k visual words
    codebook = {
        'corpus_filepath': args.corpus_filepath,
        'codebook_size': args.codebook_size,
        'max_iter': args.max_iter,
        'epsilon': args.epsilon,
        'pages': corpus['pages'],
        'codewords': list()
    }

    # assign a codeword to each cluster
    for group in range(len(d)):
        print(f'Group: {group}')
        codeword_centroid = d[group]

        # find the keypoints that belong to the current cluster and group them in pages
        codeword_keypoints = dict.fromkeys(codebook['pages'], list())
        codeword_descriptors = dict.fromkeys(codebook['pages'], list())

        curr_page_start_index = curr_page_last_index = 0
        for i, page_filename in enumerate(corpus['pages']):
            assert len(corpus['keypoints'][i]) == len(corpus['descriptors'][i])
            curr_page_last_index = curr_page_start_index + len(corpus['keypoints'][i])

            curr_page_keypoints = corpus_keypoints_vstack[curr_page_start_index:curr_page_last_index]
            curr_page_curr_group_keypoints = curr_page_keypoints[labels.ravel()[curr_page_start_index:
                                                                                curr_page_last_index] == group]
            codeword_keypoints[page_filename] = curr_page_curr_group_keypoints

            curr_page_descriptors = corpus_descriptors_vstack[curr_page_start_index:curr_page_last_index]
            curr_page_curr_group_descriptors = curr_page_descriptors[labels.ravel()[curr_page_start_index:
                                                                                    curr_page_last_index] == group]
            codeword_descriptors[page_filename] = curr_page_curr_group_descriptors

            curr_page_start_index = curr_page_last_index

        codeword = {'d': codeword_centroid, 'keypoints': codeword_keypoints, 'descriptors': codeword_descriptors}
        codebook['codewords'].append(codeword)

    assert len(codebook['codewords']) == args.codebook_size
    assert len(corpus_keypoints_vstack) == len(corpus_descriptors_vstack)
    assert sum([len(v) for codeword in codebook['codewords']
                for v in codeword['keypoints'].values()]) == len(corpus_keypoints_vstack)
    assert sum([len(v) for codeword in codebook['codewords']
                for v in codeword['descriptors'].values()]) == len(corpus_descriptors_vstack)

    utils.save_codebook(codebook, codebook_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_filepath')
    parser.add_argument('codebook_size', type=int)
    parser.add_argument('max_iter', type=int)
    parser.add_argument('epsilon', type=float)
    args = parser.parse_args()
    run()
