import os
import numpy as np
import OneShotLearningModel
import face_detect
import model_utils
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


folderpath = os.path.dirname(os.path.abspath(__file__))
nn4_small2_pretrained = OneShotLearningModel.create_model()
nn4_small2_pretrained.load_weights('modelCheckPoint\\facenet_nn4\\nn4.h5')


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file
        self.image = None

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        # Check file extension. Allow only jpg/jpeg' files.
        f = path + '\\' + i
        ext = os.path.splitext(f)[1]
        name = str(i.split('.')[0])
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata(path, name, f))
    return np.array(metadata)


def preprocess_image(image):
    # image = cv2.resize(image, (160, 64))
    image = np.array(image)

    image = cv2.resize(image, (96, 96))

    # mean, std = image.mean(), image.std()
    # image = ((image - mean) / std).astype(np.float32)
    image = (image / 255.).astype(np.float32)

    return image


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def main():
    testpath = folderpath + '\\RecognitionDemo\\test1'
    testimagepath = [testpath + '\\' + i for i in os.listdir(testpath)]
    datapath = folderpath + '\\data\\dataset\\FaceDataset50_2'
    metadata = load_metadata(datapath)

    for imagepath in testimagepath:
        embedded = np.zeros((metadata.shape[0] + 1, 128))
        imagetest = cv2.imread(imagepath)
        faces = face_detect.facebounding(imagetest)
        faceitem = faces[0]
        face = faceitem.image
        testface = preprocess_image(face)
        embedded[0] = nn4_small2_pretrained.predict(np.expand_dims(testface, axis=0))[0]

        for i in range(metadata.shape[0]):
            imageItem = metadata[i]
            image = cv2.imread(imageItem.file)
            imageItem.image = image
            img = preprocess_image(image)
            # # scale RGB values to interval [0,1]
            # img = (img / 255.).astype(np.float32)
            # # obtain embedding vector for image
            embedded[i+1] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


        imgDistanceList = []
        for i in range(len(embedded)-1):
            imgDistanceList.append(distance(embedded[0], embedded[i+1]))

        # for i in range(len(embedded)-1):
        #     print(metadata[i].file)
        #     print(imgDistanceList[i])
        print()
        minIndex = imgDistanceList.index(min(imgDistanceList))
        print(metadata[minIndex].file)
        print(min(imgDistanceList))
        cv2.imshow('1', metadata[minIndex].image)
        cv2.imshow('2', testface)
        cv2.waitKey()

if __name__ == '__main__':
    main()