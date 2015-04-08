#!/usr/bin/env python2 -u
import collections
import cPickle
import cv2
import json
import re
import sys
import utils


KeyPoint = collections.namedtuple('KeyPoint', 'pt angle size response')

def cv2_to_namedtuple_keypoints(keypoints):
    return[KeyPoint(keypoint.pt, keypoint.angle, keypoint.size, keypoint.response) for keypoint in keypoints]

def tuple_to_namedtuple_keypoints(keypoints):
    return[KeyPoint._make(keypoint) for keypoint in keypoints]

def matches_to_namedtuple_keypoints(keypoints):
    return[None if not keypoint else KeyPoint._make(keypoint) for keypoint in keypoints]

################################################################################

def save_corpus(corpus, corpus_file_path):
    for i, page_keypoints in enumerate(corpus['keypoints']):
        corpus['keypoints'][i] = cv2_to_namedtuple_keypoints(page_keypoints)

    with open(corpus_file_path, 'wb') as f:
        cPickle.dump(corpus, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_corpus(corpus_file_path):
    with open(corpus_file_path, 'rb') as f:
        corpus = cPickle.load(f)

    corpus['keypoints'] = [tuple_to_namedtuple_keypoints(page_keypoints)
                           for page_keypoints in corpus['keypoints']]
    return corpus


def save_codebook(codebook, codebook_file_path):
    with open(codebook_file_path, 'wb') as f:
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_codebook(codebook_file_path):
    with open(codebook_file_path, 'rb') as f:
        codebook = cPickle.load(f)

    for codeword in codebook['codewords']:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = tuple_to_namedtuple_keypoints(codeword['keypoints'][page])
    return codebook


def save_matches(matches, matches_file_path):
    with open(matches_file_path, 'w') as f:
        json.dump(matches, f, indent=4)

def load_matches(matches_file_path):
    with open(matches_file_path) as f:
        matches = json.load(f)

    for page in matches:
        for i, match_scored in enumerate(matches[page]):
            match = match_scored[0]
            score = match_scored[1]
            matches[page][i] = (matches_to_namedtuple_keypoints(match), score)
    return matches

################################################################################

def pcr():
    if len(sys.argv) != 3:
        sys.exit('Usage: {0} pcr corpus_file_path'.format(sys.argv[0]))

    corpus_file_path = sys.argv[2]
    print 'Starting pcr...'
    print '   {0: <16} = {1}'.format('corpus_file_path', corpus_file_path)

    ########################################################################

    print 'Corpus info:'
    corpus = utils.load_corpus(corpus_file_path)

    print '   {0: <20} = {1}'.format('pages_directory_path', corpus['pages_directory_path'])
    print '   {0: <20} = {1}'.format('contrast_threshold', corpus['contrast_threshold'])
    print '   {0: <20} = {1}'.format('n_octave_layers', corpus['n_octave_layers'])
    print '   {0: <20} = {1}'.format('pages', corpus['pages'])
    print '   {0: <20} = {1}'.format('keypoints per page', [len(page_keypoints) for page_keypoints in corpus['keypoints']])
    print '   {0: <20} = {1}'.format('descriptors per page', [len(page_descriptors) for page_descriptors in corpus['descriptors']])

def pcd():
    if len(sys.argv) != 3:
        sys.exit('Usage: {0} pcd codebook_file_path'.format(sys.argv[0]))

    codebook_file_path = sys.argv[2]
    print 'Starting pcd...'
    print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)

    ########################################################################

    print 'Codebook info:'
    codebook = load_codebook(codebook_file_path)

    print '   {0: <23} = {1}'.format('corpus_file_path', codebook['corpus_file_path'])
    print '   {0: <23} = {1}'.format('codebook_size', codebook['codebook_size'])
    print '   {0: <23} = {1}'.format('max_iter', codebook['max_iter'])
    print '   {0: <23} = {1}'.format('epsilon', codebook['epsilon'])
    print '   {0: <23} = {1}'.format('num of codewords', len(codebook['codewords']))
    print '   {0: <23} = {1}'.format('keypoints count', sum([len(v) for codeword in codebook['codewords']
                                                             for v in codeword['keypoints'].viewvalues()]))
    print '   {0: <23} = {1}'.format('keypoints per codewords', [len(v) for codeword in codebook['codewords']
                                                                 for v in codeword['keypoints'].viewvalues()])

def pmt():
    if len(sys.argv) != 3:
        sys.exit('Usage: {0} pmt matches_file_path'.format(sys.argv[0]))

    matches_file_path = sys.argv[2]
    print 'Starting pmt...'
    print '   {0: <17} = {1}'.format('matches_file_path', matches_file_path)

    ########################################################################

    print 'Matches info:'
    matches = load_matches(matches_file_path)

    print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)
    print '   {0: <18} = {1}'.format('pages', matches.keys())

################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: {0} mode'.format(sys.argv[0]))

    mode = sys.argv[1]
    locals()[mode]()
