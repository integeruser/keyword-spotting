#!/usr/bin/env python2
import argparse
import os

import cv2
import utils


def run(args, output_directory_path="", force=False):
    pages_directory_path = args["pages_directory_path"]
    matches_file_path = args["matches_file_path"]

    print("Extracting matches...")
    print("   {:<21} = \"{}\"".format("pages_directory_path", pages_directory_path))
    print("   {:<21} = \"{}\"".format("matches_file_path", matches_file_path))

    # load matches
    print("Loading matches...")
    matches = utils.load_matches(matches_file_path)

    # extract top-k matches
    k = 20
    print("Extracting top-{0} matches in each page...".format(k))
    for page, page_matches in matches.items():
        top_k_page_matches = sorted(page_matches, key=lambda x: x[1])[:k]
        assert len(top_k_page_matches) <= k

        page_image = cv2.imread("{0}/{1}".format(pages_directory_path, page))
        for i, match_scored in enumerate(top_k_page_matches):
            match = match_scored[0]

            # keep only non-empy keypoints
            match_keypoints = [keypoint for keypoint in match if keypoint]
            match_keypoints = utils.namedtuple_keypoints_to_cv2(match_keypoints)

            # draw keypoints on original image
            print(len(match_keypoints))
            page_image = cv2.drawKeypoints(page_image, match_keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # draw box around match
            offset_pixel = 25
            x_min = float("inf")
            y_min = float("inf")
            x_max = 0
            y_max = 0
            for keypoint in match_keypoints:
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            x_min += - offset_pixel
            y_min += - offset_pixel
            x_max += + offset_pixel
            y_max += + offset_pixel
            cv2.rectangle(page_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imwrite("{0}/{1}.png".format(output_directory_path, page), page_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", default="")
    parser.add_argument("-f", action="store_true", default=False)
    parser.add_argument("pages_directory_path")
    parser.add_argument("matches_file_path")
    args = parser.parse_args()
    run(vars(args), output_directory_path=args.o, force=args.f)
