import cv2


def draw_keypoints(image, keypoints, output_file):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints)
    cv2.imwrite(output_file, image_with_keypoints)
