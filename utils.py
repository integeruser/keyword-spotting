#!/usr/bin/env python2 -u
import copy
import cPickle
import cv2
import sys
import utils


def serialize_keypoints(keypoints):
    return [(keypoint.pt, keypoint.size,
             keypoint.angle, keypoint.response,
             keypoint.octave, keypoint.class_id) for keypoint in keypoints]


def deserialize_keypoints(keypoints):
    return [cv2.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], _size=keypoint[1],
                         _angle=keypoint[2], _response=keypoint[3],
                         _octave=keypoint[4], _class_id=keypoint[5]) for keypoint in keypoints]

################################################################################

def save_corpus(corpus, corpus_file_path):
    corpus = copy.deepcopy(corpus)

    for i, page_keypoints in enumerate(corpus['keypoints']):
        corpus['keypoints'][i] = serialize_keypoints(page_keypoints)

    with open(corpus_file_path, 'wb') as f:
        cPickle.dump(corpus, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_corpus(corpus_file_path):
    with open(corpus_file_path, 'rb') as f:
        corpus = cPickle.load(f)

    corpus['keypoints'] = [
        deserialize_keypoints(page_keypoints) for page_keypoints in corpus['keypoints']]
    return corpus


def save_codebook(codebook, codebook_file_path):
    codebook = copy.deepcopy(codebook)

    for codeword in codebook['codewords']:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = serialize_keypoints(codeword['keypoints'][page])

    with open(codebook_file_path, 'wb') as f:
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_codebook(codebook_file_path):
    with open(codebook_file_path, 'rb') as f:
        codebook = cPickle.load(f)

    for codeword in codebook['codewords']:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = deserialize_keypoints(codeword['keypoints'][page])
    return codebook


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

    print '   {0: <16} = {1}'.format('corpus_file_path', codebook['corpus_file_path'])
    print '   {0: <16} = {1}'.format('codebook_size', codebook['codebook_size'])
    print '   {0: <16} = {1}'.format('max_iter', codebook['max_iter'])
    print '   {0: <16} = {1}'.format('epsilon', codebook['epsilon'])
    print '   {0: <16} = {1}'.format('num of codewords', len(codebook['codewords']))

################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: {0} mode'.format(sys.argv[0]))

    mode = sys.argv[1]
    locals()[mode]()
