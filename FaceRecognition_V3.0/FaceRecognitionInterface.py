from face_recognition import preprocess_image, distance
from face_detect import facebounding
import OneShotLearningModel
import cv2
import os
import numpy as np
import threading
import sys

Personal_ID = ''
phone_number = ''
pass_state = ''

nn4_small2_pretrained = OneShotLearningModel.create_model()
nn4_small2_pretrained.load_weights('modelCheckPoint\\facenet_nn4\\nn4.h5')

def input_imformation():
    global Personal_ID
    global phone_number
    global pass_state
    Personal_ID = ''
    phone_number = ''
    pass_state = ''
    print('Please input your phone number:')
    phone_number = input()
    print('Please input your GIN/FIN:')
    Personal_ID = input()
    print(phone_number+'----'+Personal_ID)
    pass_state = 'pass'


def get_feature1():
    global faceitem
    capture = cv2.VideoCapture(0)
    cv2.namedWindow('frame', 2)
    cv2.namedWindow('face', 2)
    P1 = threading.Thread(target=input_imformation, args=())
    P1.start()
    while True:
        if not P1.is_alive():
            P1 = threading.Thread(target=input_imformation, args=())
            P1.start()
        ret, frame = capture.read()
        image = np.array(frame)
        cv2.imshow('frame', frame)
        faces = facebounding(image)
        try:
            faceitem = faces[0]
            cv2.imshow('face', faceitem.image)
        except:
            cv2.imshow('face', image)
        if pass_state == 'pass':
            P3 = threading.Thread(target=get_feature, args=())
            P3.start()
            P3.join()
        key = cv2.waitKey(1)
        if key == ord('q'):  # 判断是哪一个键按下
            cv2.destroyAllWindows()
            break


def get_feature():
    global path
    global faceitem
    global Personal_ID
    global pass_state
    if pass_state == 'pass':
        csvPath = path + '\\Facedataset.csv'
        cavepath = path + '\\Facedataset\\' + Personal_ID + '.jpg'
        cv2.imwrite(cavepath, faceitem.image)
        with open(csvPath, 'a', encoding='utf-8') as f:
            faceimg = faceitem.image
            testface = preprocess_image(faceimg)
            face_feature = nn4_small2_pretrained.predict(np.expand_dims(testface, axis=0))[0]
            face_feature = list(map(str, face_feature))
            face_feature = ','.join(face_feature)
            logs = Personal_ID + ',' + phone_number + ',' + face_feature + '\n'
            f.writelines(logs)
            print('Pass')
        f.close()
        pass_state = None


def get_face_and_ID_Interface():
    while True:
        global Personal_ID
        global phone_number
        global pass_state

        P2 = threading.Thread(target=get_feature1, args=())
        P2.start()
        P2.join()


def Face_Recognition_Interface(image):
    dataset_path = 'data\\dataset\\FaceDataset50_1.csv'
    label_ID = []
    phone = []
    with open(dataset_path, encoding='utf-8') as f:
        reader = f.readlines()
        embedded = np.zeros((len(reader) + 1, 128))
        i = 1
        for row in reader:
            label_ID.append(row.split(',')[0])
            phone.append(row.split(',')[1])
            feature = np.array(list(map(float, row.split(',')[2:])))
            embedded[i, :] = feature
            i += 1
    f.close()

    faces = facebounding(image)


    faceitem = faces[0]
    faceimage = faceitem.image
    faceimage = preprocess_image(faceimage)
    embedded[0] = nn4_small2_pretrained.predict(np.expand_dims(faceimage, axis=0))[0]

    imgDistanceList = []
    for i in range(len(embedded) - 1):
        imgDistanceList.append(distance(embedded[0], embedded[i + 1]))
    minIndex = imgDistanceList.index(min(imgDistanceList))
    print(label_ID[minIndex])
    print(phone[minIndex])
    print(min(imgDistanceList))
    imageSet_path = dataset_path.split('.')[0]
    image_path = [imageSet_path + '\\' + i for i in os.listdir(imageSet_path)]
    for i in image_path:
        if i.split('\\')[-1].split('_')[0] == label_ID[minIndex]:
            targret = i
            print(targret)
    image_target = cv2.imread(targret)
    cv2.imshow('1', image_target)
    cv2.imshow('2', image)
    cv2.waitKey(10)


if __name__ == '__main__':

    # a = 'E:\\Project\\FaceRecognition_V2.0\\RecognitionDemo\\train_13_B3_00106.jpg'
    # image = cv2.imread(a)
    # Face_Recognition_Interface(image)
    path = sys.argv[1]
    if not os.path.exists(path + '\\Facedataset'):
        os.mkdir(path + '\\Facedataset')
    get_face_and_ID_Interface()
