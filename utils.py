import cv2


def serialize_keypoints(keypoints):
    return [(keypoint.pt, keypoint.size,
             keypoint.angle, keypoint.response,
             keypoint.octave, keypoint.class_id) for keypoint in keypoints]


def deserialize_keypoints(keypoints):
    return [cv2.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], _size=keypoint[1],
                         _angle=keypoint[2], _response=keypoint[3],
                         _octave=keypoint[4], _class_id=keypoint[5]) for keypoint in keypoints]

################################################################################
