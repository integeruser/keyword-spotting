#!/usr/bin/env python2 -u
import sys
import utils


if len(sys.argv) != 2:
    sys.exit('Usage: {0} codebook_file_path'.format(sys.argv[0]))

codebook_file_path = sys.argv[1]
print 'Starting script...'
print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)

################################################################################

print 'Codebook info:'
codebook = utils.load_codebook(codebook_file_path)

print '   {0: <16} = {1}'.format('corpus_file_path', codebook['corpus_file_path'])
print '   {0: <16} = {1}'.format('codebook_size', codebook['codebook_size'])
print '   {0: <16} = {1}'.format('max_iter', codebook['max_iter'])
print '   {0: <16} = {1}'.format('epsilon', codebook['epsilon'])
print '   {0: <16} = {1}'.format('num of codewords', len(codebook['codewords']))
