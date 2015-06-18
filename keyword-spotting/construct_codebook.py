#!/usr/bin/env python2
import argparse
import hashlib
import itertools
import numpy
import os

import cv2
import utils


def run(args, output_directory_path="", force=False):
    corpus_file_path = args["corpus_file_path"]
    codebook_size = int(args["codebook_size"])
    max_iter = int(args["max_iter"])
    epsilon = float(args["epsilon"])

    print("Constructing codebook...")
    print("   {:<16} = \"{}\"".format("corpus_file_path", corpus_file_path))
    print("   {:<16} = {}".format("codebook_size", codebook_size))
    print("   {:<16} = {}".format("max_iter", max_iter))
    print("   {:<16} = {}".format("epsilon", epsilon))

    # compute codebook fingerprint, then check if the same codebook was already constructed
    fingerprint = os.path.basename(corpus_file_path)
    fingerprint += str(codebook_size)
    fingerprint += str(max_iter)
    fingerprint += str(epsilon)

    codebook_file_name = "codebook-{}".format(hashlib.sha256(fingerprint).hexdigest()[:7])
    codebook_file_path = os.path.join(output_directory_path, codebook_file_name)
    if not force and os.path.isfile(codebook_file_path):
        print("Codebook already exists and therefore construction is skipped. Use -f to reconstruct it.")
    else:
        # construct codebook
        # load precomputed keypoints and descriptors
        print("Loading corpus...")
        corpus = utils.load_corpus(corpus_file_path)
        assert len(corpus["pages"]) > 0
        assert len(corpus["pages"]) == len(corpus["keypoints"]) == len(corpus["descriptors"])
        for page_keypoints, page_descriptors in itertools.izip(corpus["keypoints"], corpus["descriptors"]):
            assert len(page_keypoints) == len(page_descriptors)

        # run k-means on the descriptors space
        print("Running k-means on the descriptors space...")
        corpus_descriptors_vstack = numpy.vstack(corpus["descriptors"])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        attempts = 10
        compactness, labels, d = cv2.kmeans(corpus_descriptors_vstack, codebook_size,
                                            criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        assert len(d) == codebook_size

        # construct the codebook of k visual words
        print("Constructing codebook...")
        corpus_keypoints_vstack = numpy.asarray(
            [keypoint for page_keypoints in corpus["keypoints"] for keypoint in page_keypoints])

        codebook = {
            "corpus_file_path": corpus_file_path,
            "codebook_size": codebook_size,
            "max_iter": max_iter,
            "epsilon": epsilon,
            "pages": corpus["pages"],
            "codewords": list()
        }

        # create a codeword for each k-means group found
        for group in range(len(d)):
            codeword_centroid = d[group]

            # find keypoints that belong to the current group, and group them in pages
            codeword_keypoints = dict.fromkeys(codebook["pages"], list())
            codeword_descriptors = dict.fromkeys(codebook["pages"], list())

            curr_page_start_index = curr_page_last_index = 0
            for i, page in enumerate(corpus["pages"]):
                assert len(corpus["keypoints"][i]) == len(corpus["descriptors"][i])
                curr_page_last_index = curr_page_start_index + len(corpus["keypoints"][i])

                curr_page_keypoints = corpus_keypoints_vstack[curr_page_start_index:curr_page_last_index]
                curr_page_curr_group_keypoints = curr_page_keypoints[
                    labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
                codeword_keypoints[page] = curr_page_curr_group_keypoints

                curr_page_descriptors = corpus_descriptors_vstack[curr_page_start_index:curr_page_last_index]
                curr_page_curr_group_descriptors = curr_page_descriptors[
                    labels.ravel()[curr_page_start_index:curr_page_last_index] == group]
                codeword_descriptors[page] = curr_page_curr_group_descriptors

                curr_page_start_index = curr_page_last_index

            codeword = {
                "d": codeword_centroid,
                "keypoints": codeword_keypoints,
                "descriptors": codeword_descriptors
            }
            codebook["codewords"].append(codeword)

        assert len(codebook["codewords"]) == codebook_size
        assert len(corpus_keypoints_vstack) == len(corpus_descriptors_vstack)
        assert sum([len(v) for codeword in codebook["codewords"]
                    for v in codeword["keypoints"].viewvalues()]) == len(corpus_keypoints_vstack)
        assert sum([len(v) for codeword in codebook["codewords"]
                    for v in codeword["descriptors"].viewvalues()]) == len(corpus_descriptors_vstack)

        # save codebook
        print("Saving {}...".format(codebook_file_path))
        utils.save_codebook(codebook, codebook_file_path)  # codebook is modified after this operation
    return codebook_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", default="")
    parser.add_argument("-f", action="store_true", default=False)
    parser.add_argument("corpus_file_path")
    parser.add_argument("codebook_size", type=int)
    parser.add_argument("max_iter", type=int)
    parser.add_argument("epsilon", type=float)
    args = parser.parse_args()
    run(vars(args), output_directory_path=args.o, force=args.f)
