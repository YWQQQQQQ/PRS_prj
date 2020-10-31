import cv2
from face_detect import facebounding
from face_recognition import preprocess_image
import OneShotLearningModel
import os
import numpy as np
import sys


nn4_small2_pretrained = OneShotLearningModel.create_model()
nn4_small2_pretrained.load_weights('modelCheckPoint\\facenet_nn4\\nn4.h5')

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def face_rec(num):
    dataset_path = 'data\\dataset\\test\\test'+str(num)+'.csv'
    label_ID = []
    with open(dataset_path, encoding='utf-8') as f:
        reader = f.readlines()
        embedded = np.zeros((len(reader) + 1, 128))
        i = 1
        for row in reader:
            label_ID.append(row.split(',')[0])
            feature = np.array(list(map(float, row.split(',')[1:])))
            embedded[i, :] = feature
            i += 1
    f.close()
    path = 'data\\dataset\\Yuchao'
    imganchorpath = [path + '\\' + i for i in os.listdir(path)]
    for imganchor in imganchorpath:
        image1 = cv2.imread(imganchor)
        faces = facebounding(image1)
        size = []
        for i in faces:
            h, w, c = i.image.shape
            size.append(h)
        index = size.index(max(size))
        image1 = preprocess_image(faces[index].image)
        embedded[0] = nn4_small2_pretrained.predict(np.expand_dims(image1, axis=0))[0]

        imgDistanceList = []
        for i in range(len(embedded) - 1):
            imgDistanceList.append(distance(embedded[0], embedded[i + 1]))
        minIndex = imgDistanceList.index(min(imgDistanceList))
        cv2.rectangle(faces[index].container_image,
                                    (faces[index].bounding_box[0], faces[index].bounding_box[1]),
                                    (faces[index].bounding_box[2], faces[index].bounding_box[3]),
                                    (0, 155, 255), 5)
        text = label_ID[minIndex]
        cv2.putText(faces[index].container_image, text, (faces[index].bounding_box[0], faces[index].bounding_box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 155, 0), 1)
        print(label_ID[minIndex])
        print(min(imgDistanceList))
        # cv2.namedWindow('image', 2)
        # cv2.namedWindow('result', 2)
        cv2.imshow(imganchor.split('\\')[-1], faces[0].container_image)
        cv2.imshow('result_'+imganchor.split('\\')[-1], cv2.imread('data\\dataset\\test\\test'+str(num)+'\\'+label_ID[minIndex]))
        cv2.waitKey()

def video_test():
    capture = cv2.VideoCapture('data\\dataset\\yuchao.mp4')
    cv2.namedWindow('frame', 2)
    cv2.namedWindow('face', 2)
    dataset_path = 'data\\dataset\\test\\test100.csv'
    label_ID = []
    with open(dataset_path, encoding='utf-8') as f:
        reader = f.readlines()
        embedded = np.zeros((len(reader) + 1, 128))
        i = 1
        for row in reader:
            label_ID.append(row.split(',')[0])
            feature = np.array(list(map(float, row.split(',')[1:])))
            embedded[i, :] = feature
            i += 1
    f.close()
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        image = np.array(frame)
        faces = facebounding(image)
        text = ''
        try:
            faceitem = faces[0]
            image = faceitem.image
            cv2.imshow('face', image)
            image1 = preprocess_image(image)
            embedded[0] = nn4_small2_pretrained.predict(np.expand_dims(image1, axis=0))[0]
            imgDistanceList = []
            for i in range(len(embedded) - 1):
                imgDistanceList.append(distance(embedded[0], embedded[i + 1]))
            minIndex = imgDistanceList.index(min(imgDistanceList))
            if float(min(imgDistanceList)) < 0.23:
                print(label_ID[minIndex])
                print(min(imgDistanceList))
                text = label_ID[minIndex]
                cv2.rectangle(frame,
                              (faces[0].bounding_box[0], faces[0].bounding_box[1]),
                              (faces[0].bounding_box[2], faces[0].bounding_box[3]),
                              (0, 155, 255), 2)
                cv2.putText(frame, text, (faces[0].bounding_box[0], faces[0].bounding_box[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 155, 255), 2)
                cv2.imshow('frame', frame)
                cv2.imshow('person', cv2.imread('data\\dataset\\test\\test100\\' + label_ID[minIndex]))
                cv2.waitKey(5)
        except:
            cv2.imshow('face', image)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # 判断是哪一个键按下
            cv2.destroyAllWindows()
            break
def main():
    num = sys.argv[1]
    face_rec(num)

    # video_test()



if __name__ == '__main__':
    main()
