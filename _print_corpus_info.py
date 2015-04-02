#!/usr/bin/env python2 -u
import sys
import utils


if len(sys.argv) != 2:
    sys.exit('Usage: {0} corpus_file_path'.format(sys.argv[0]))

corpus_file_path = sys.argv[1]
print 'Starting script...'
print '   {0: <16} = {1}'.format('corpus_file_path', corpus_file_path)

################################################################################

print 'Corpus info:'
corpus = utils.load_corpus(corpus_file_path)

print '   {0: <20} = {1}'.format('pages_directory_path', corpus['pages_directory_path'])
print '   {0: <20} = {1}'.format('contrast_threshold', corpus['contrast_threshold'])
print '   {0: <20} = {1}'.format('n_octave_layers', corpus['n_octave_layers'])
print '   {0: <20} = {1}'.format('pages', corpus['pages'])
print '   {0: <20} = {1}'.format('keypoints per page', [len(page_keypoints) for page_keypoints in corpus['keypoints']])
print '   {0: <20} = {1}'.format('descriptors per page', [len(page_descriptors) for page_descriptors in corpus['descriptors']])
