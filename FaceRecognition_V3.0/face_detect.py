import os
import cv2
import numpy as np
import tensorflow as tf
from model.mtcnnModel import PNet, RNet, ONet
from PIL import Image, ImageDraw
from model.utils import detect_face
import imageio


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)

tf.get_logger().setLevel('ERROR')
print("Tensorflow version: ", tf.__version__)
print(tf.test.gpu_device_name())


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.keypoint = None


def load_weights(model, weights_file):
    weights_dict = np.load(weights_file, encoding='latin1',allow_pickle=True).item()
    for layer_name in weights_dict.keys():
        layer = model.get_layer(layer_name)
        if "conv" in layer_name:
            layer.set_weights([weights_dict[layer_name]["weights"], weights_dict[layer_name]["biases"]])
        else:
            prelu_weight = weights_dict[layer_name]['alpha']
            try:
                layer.set_weights([prelu_weight])
            except:
                layer.set_weights([prelu_weight[np.newaxis, np.newaxis, :]])
    return True


def facebounding(image):
    total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)
    faces = []
    for bounding_box, keypoints in zip(total_boxes, points.T):
        face = Face()
        bounding_boxes = {
            'box': [int(bounding_box[0]), int(bounding_box[1]),
                    int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])],
            'confidence': bounding_box[-1],
            'keypoints': {
                'left_eye': (int(keypoints[0]), int(keypoints[5])),
                'right_eye': (int(keypoints[1]), int(keypoints[6])),
                'nose': (int(keypoints[2]), int(keypoints[7])),
                'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                'mouth_right': (int(keypoints[4]), int(keypoints[9])),
            }
        }
        bounding_box = bounding_boxes['box']
        keypoints = bounding_boxes['keypoints']
        # cv2.rectangle(image,
        #               (bounding_box[0], bounding_box[1]),
        #               (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
        #               (0, 155, 255), 2)
        face.keypoint = keypoints
        face.container_image = image
        face.bounding_box = np.zeros(4, dtype=np.int32)
        facecenter = np.array((bounding_box[0] + int(bounding_box[2]/2), bounding_box[1] + int(bounding_box[3]/2)))
        halfLenth = int((bounding_box[2]+bounding_box[3])/4)
        move = int(halfLenth*0.2)
        face.bounding_box[0] = facecenter[0] - halfLenth
        face.bounding_box[1] = facecenter[1] - halfLenth + move
        face.bounding_box[2] = facecenter[0] + halfLenth
        face.bounding_box[3] = facecenter[1] + halfLenth + move
        face.image = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        faces.append(face)

    return faces


folderpath = os.path.dirname(os.path.abspath(__file__))
pnet, rnet, onet = PNet(), RNet(), ONet()

pnet(tf.ones(shape=[1,  12,  12, 3]))
rnet(tf.ones(shape=[1,  24,  24, 3]))
onet(tf.ones(shape=[1,  48,  48, 3]))
load_weights(pnet, folderpath+"\\modelCheckPoint\\mtcnn\\det1.npy")
load_weights(rnet, folderpath+"\\modelCheckPoint\\mtcnn\\det2.npy")
load_weights(onet, folderpath+"\\modelCheckPoint\\mtcnn\\det3.npy")