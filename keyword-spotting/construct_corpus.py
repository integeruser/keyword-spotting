#!/usr/bin/env python3
import argparse
import hashlib
import os

import cv2
import utils


def run(args, output_directory_path="", force=False):
    pages_directory_path = args["pages_directory_path"]
    contrast_threshold = float(args["contrast_threshold"])
    n_octave_layers = int(args["n_octave_layers"])

    print("Constructing corpus...")
    print("   {:<20} = \"{}\"".format("pages_directory_path", pages_directory_path))
    print("   {:<20} = {}".format("contrast_threshold", contrast_threshold))
    print("   {:<20} = {}".format("n_octave_layers", n_octave_layers))

    # compute corpus fingerprint, then check if the same corpus was already constructed
    fingerprint = "".join([page_file_name for page_file_name in os.listdir(pages_directory_path)])
    fingerprint += str(contrast_threshold)
    fingerprint += str(n_octave_layers)
    fingerprint = fingerprint.encode("utf-8")

    corpus_file_name = "corpus-{}".format(hashlib.sha256(fingerprint).hexdigest()[:7])
    corpus_file_path = os.path.join(output_directory_path, corpus_file_name)
    if not force and os.path.isfile(corpus_file_path):
        print("Corpus already exists and therefore construction is skipped. Use -f to reconstruct it.")
    else:
        # construct corpus
        # load each page as grey scale, detect its keypoints and compute its descriptors
        print("Loading pages...")
        corpus = {
            "pages_directory_path": pages_directory_path,
            "contrast_threshold": contrast_threshold,
            "n_octave_layers": n_octave_layers,
            "pages": list(),
            "keypoints": list(),
            "descriptors": list()
        }

        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)

        for page_file_name in os.listdir(pages_directory_path):
            print("   Detecting keypoints and computing descriptors on \"{}\"...".format(page_file_name))
            page_image = cv2.imread(os.path.join(pages_directory_path, page_file_name))
            page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

            page_keypoints, page_descriptors = sift.detectAndCompute(page_image, None)
            assert len(page_keypoints) > 0
            assert len(page_keypoints) == len(page_descriptors)

            corpus["pages"].append(page_file_name)
            corpus["keypoints"].append(page_keypoints)
            corpus["descriptors"].append(page_descriptors)

        # save pages, keypoints and descriptors to file
        print("Saving {}...".format(corpus_file_path))
        utils.save_corpus(corpus, corpus_file_path)  # corpus is modified after this operation
    return corpus_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", default="", help="")
    parser.add_argument("-f", action="store_true", default=False, help="")
    parser.add_argument("pages_directory_path", help="")
    parser.add_argument("contrast_threshold", type=float, help="")
    parser.add_argument("n_octave_layers", type=int, help="")
    args = parser.parse_args()
    run(vars(args), output_directory_path=args.o, force=args.f)
