#!/usr/bin/env python2
import collections
import cPickle
import json
import os
import re
import sys

import cv2


KeyPoint = collections.namedtuple('KeyPoint', 'pt size angle response octave')


def cv2_to_namedtuple_keypoints(keypoints):
    return [KeyPoint(keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave)
            for keypoint in keypoints]

def namedtuple_keypoints_to_cv2(keypoints):
    return [cv2.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], _size=keypoint[1],
                         _angle=keypoint[2], _response=keypoint[3], _octave=keypoint[4])
            for keypoint in keypoints]

def tuple_to_namedtuple_keypoints(keypoints):
    return [KeyPoint._make(keypoint) for keypoint in keypoints]

def matches_to_namedtuple_keypoints(keypoints):
    return [KeyPoint._make(keypoint) if keypoint else None for keypoint in keypoints]

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

def dcr():
    if len(sys.argv) != 4:
        sys.exit('Usage: {0} dcr corpus_file_path page_file_path'.format(sys.argv[0]))

    corpus_file_path = sys.argv[2]
    page_file_path = sys.argv[3]
    print 'Starting dcr...'
    print '   {0: <16} = {1}'.format('corpus_file_path', corpus_file_path)
    print '   {0: <16} = {1}'.format('page_file_path', page_file_path)

    ########################################################################

    page_image = cv2.imread(page_file_path)

    corpus = load_corpus(corpus_file_path)
    page = os.path.basename(page_file_path)
    i = corpus['pages'].index(page)
    page_keypoints = namedtuple_keypoints_to_cv2(corpus['keypoints'][i])

    cv2.imshow('page', cv2.drawKeypoints(page_image, page_keypoints, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    cv2.waitKey()


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

def codeword_vs_query():
    if len(sys.argv) != 4:
        sys.exit('Usage: {0} codeword_vs_query pages_directory_path query_file_path'.format(
            sys.argv[0]))

    pages_directory_path = sys.argv[2]
    query_file_path = sys.argv[3]
    print 'Starting script...'
    print '   {0: <19} = {1}'.format('pages_directory_path', pages_directory_path)
    print '   {0: <19} = {1}'.format('query_file_path', query_file_path)

    ########################################################################

    with open('debug.json') as f:
        debug = json.load(f)

    query_image = cv2.imread(query_file_path)

    page_index = 0
    query_keypoint_index = 0
    while True:
        page = debug.keys()[page_index]

        query_keypoint = debug[page][query_keypoint_index][0]
        codeword_keypoints = debug[page][query_keypoint_index][1]

        page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))

        cv2.imshow('codeword-keypoints{0}-{1}'.format(page_index, query_keypoint_index),
                   cv2.drawKeypoints(page_image, utils.namedtuple_keypoints_to_cv2(codeword_keypoints),
                                     None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        cv2.moveWindow('codeword-keypoints{0}-{1}'.format(page_index, query_keypoint_index), 200, 200)

        cv2.imshow('query-keypoints{0}-{1}'.format(page_index, query_keypoint_index),
                   cv2.drawKeypoints(query_image, utils.namedtuple_keypoints_to_cv2([query_keypoint]),
                                     None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        cv2.moveWindow('query-keypoints{0}-{1}'.format(page_index, query_keypoint_index), 200, 400)
        k = cv2.waitKey()
        cv2.destroyAllWindows()

        if k == ord('j'):
            query_keypoint_index -= 1
            if query_keypoint_index == -1:
                query_keypoint_index = len(debug[page])-1
                page_index = (page_index-1) % len(debug.keys())
        elif k == ord('l'):
            query_keypoint_index += 1
            if query_keypoint_index == len(debug[page]):
                query_keypoint_index = 0
                page_index = (page_index+1) % len(debug.keys())

################################################################################

def save_debug(codebook, query_points):
    debug = dict()
    for page in codebook['pages']:
        debug[page] = list()
        for query_point in query_points:
            curr_codeword = query_point['w']
            curr_codeword_keypoints = curr_codeword['keypoints'][page]
            query_keypoint = cv2_to_namedtuple_keypoints([query_point['x']])[0]
            debug[page].append((query_keypoint, curr_codeword_keypoints))

    with open('debug.json', 'w') as f:
        json.dump(debug, f, indent=4)

################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: {0} mode'.format(sys.argv[0]))

    mode = sys.argv[1]
    locals()[mode]()
