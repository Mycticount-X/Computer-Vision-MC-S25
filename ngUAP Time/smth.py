import cv2
import math
import os
import numpy as np

TRAIN_PATH = "./images/train/"
TEST_PATH = "./images/test/"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def TrainModel ():
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()

    face_list = []
    class_list = []
    classes = os.listdir(TRAIN_PATH)

    for idx, clx in enumerate(classes):
        folder_path = os.path.join(TRAIN_PATH, clx)

        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img_ori = cv2.imread(img_path)
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

            detected_faces = classifierx.detectMultiScale(img, 1.2, 5)

            if len(detected_faces) < 1:
                continue

            for x, y, w, h in detected_faces:
                imgface = img[y:y+h, x:x+w]
                face_list.append(imgface)
                class_list.append(idx)
    
    recognizerx.train(face_list, np.array(class_list))
    return recognizerx, classes

def TestModel (recognizerx, classes):
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)

    for raw_path in os.listdir(TEST_PATH):
        img_path = os.path.join(TEST_PATH, raw_path)
        img_ori = cv2.imread(img_path)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

        detected_faces = classifierx.detectMultiScale(img, 1.2, 5)

        if len(detected_faces) < 1:
            continue

        for x, y, w, h in detected_faces:
            imgface = img[y:y+h, x:x+w]
            res, loss = recognizerx.predict(imgface)
            loss = math.floor(loss * 100) / 100

            cv2.rectangle(img_ori, (x, y), (x+w, y+h), (0, 255, 0), 1)
            text = f'{classes[res]}: {loss}'
            cv2.putText(img_ori, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
        
        cv2.imshow('Detection Result', img_ori)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = None

    while True:
        print('')
        print('Something:')
        print('1. Train')
        print('2. Test')
        print('3. Exit')
        cc = input('>> ')
        print('')

        if cc == '1':
            model = TrainModel()

        elif cc == '2':
            if model is None:
                print('Please Train the Model First')
                continue
            TestModel(*model)
        elif cc == '3':
            print('Alright, Thank You')
            break