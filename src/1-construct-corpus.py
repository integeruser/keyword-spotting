#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys

import cv2
import utils


def run():
    pages_filenames = [
        page_filename for page_filename in os.listdir(args.gw_20p_wannot_dirpath) if page_filename.endswith('.tif')
    ]

    corpus_fingerprint = ''.join(pages_filenames)
    corpus_fingerprint += str(args.contrast_threshold)
    corpus_fingerprint += str(args.n_octave_layers)
    corpus_fingerprint = corpus_fingerprint.encode('ascii')
    corpus_filepath = f'corpus-{hashlib.sha256(corpus_fingerprint).hexdigest()[:7]}.pickle'
    if os.path.isfile(corpus_filepath):
        msg = 'The corpus already exist. Overwrite?'
        overwrite = input('{} (y/N) '.format(msg)).lower() in ('y', 'yes')
        if not overwrite: sys.exit(1)

    corpus = {
        'gw_20p_wannot_dirpath': args.gw_20p_wannot_dirpath,
        'contrast_threshold': args.contrast_threshold,
        'n_octave_layers': args.n_octave_layers,
        'pages': list(),
        'keypoints': list(),
        'descriptors': list()
    }

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=args.contrast_threshold, nOctaveLayers=args.n_octave_layers)

    for page_filename in pages_filenames:
        # load the page image as grey scale, detect its keypoints and compute its descriptors
        print(f'Processing: {page_filename}')
        page_image = cv2.imread(os.path.join(args.gw_20p_wannot_dirpath, page_filename))
        page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        page_keypoints, page_descriptors = sift.detectAndCompute(page_image, None)
        assert len(page_keypoints) > 0
        assert len(page_keypoints) == len(page_descriptors)

        corpus['pages'].append(page_filename)
        corpus['keypoints'].append(utils.cv2_to_namedtuple_keypoints(page_keypoints))
        corpus['descriptors'].append(page_descriptors)

    utils.save_corpus(corpus, corpus_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gw_20p_wannot_dirpath')
    parser.add_argument('contrast_threshold', type=float)
    parser.add_argument('n_octave_layers', type=int)
    args = parser.parse_args()
    run()
