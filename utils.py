import cPickle
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


def save_corpus(corpus, corpus_file_path):
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
    for codeword in codebook:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = serialize_keypoints(codeword['keypoints'][page])

    with open(codebook_file_path, 'wb') as f:
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)


def load_codebook(codebook_file_path):
    with open(codebook_file_path, 'rb') as f:
        codebook = cPickle.load(f)

    for codeword in codebook:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = deserialize_keypoints(codeword['keypoints'][page])
    return codebook
