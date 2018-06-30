#!/usr/bin/env python3
import argparse
import os
import tempfile

import cv2
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gw_20p_wannot_dirpath')
    parser.add_argument('matches_filepath')
    args = parser.parse_args()

    matches = utils.load_matches(args.matches_filepath)

    output_dirpath = tempfile.mkdtemp()

    # extract top-k matches
    k = 20
    print(f'Extracting top-{k} matches in each page...')
    for page_filename, page_matches in matches.items():
        top_k_page_matches = sorted(page_matches, key=lambda x: x[1])[:k]
        assert len(top_k_page_matches) <= k

        page_image = cv2.imread(f'{args.gw_20p_wannot_dirpath}/{page_filename}')
        for i, match_scored in enumerate(top_k_page_matches):
            match = match_scored[0]

            # keep only non-empty keypoints
            match_keypoints = [keypoint for keypoint in match if keypoint]
            match_keypoints = utils.namedtuple_keypoints_to_cv2(match_keypoints)

            # draw keypoints on the original page
            page_image = cv2.drawKeypoints(
                page_image, match_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # draw a box around the match
            offset_pixel = 25
            x_min = float('inf')
            y_min = float('inf')
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
            x_min += -offset_pixel
            y_min += -offset_pixel
            x_max += +offset_pixel
            y_max += +offset_pixel
            cv2.rectangle(page_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        page_image_filepath = f'{output_dirpath}/{page_filename}.png'
        print(f'Saving: {page_image_filepath}')
        cv2.imwrite(page_image_filepath, page_image)
