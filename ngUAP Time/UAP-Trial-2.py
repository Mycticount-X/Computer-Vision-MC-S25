import math
import cv2
import os
import numpy as np

TRAIN_PATH = './Dataset/Train/'
TEST_PATH = './Dataset/Test/'
MODEL_PATH = './model_recognizer.xml'
CASCADE_PATH = './haarcascade_frontalface_default.xml'

def TrainTest ():
    print('Initialize Training...\n')

    # Init
    cascaderx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()
    
    face_list = []
    class_list = []
    classes = os.listdir(TRAIN_PATH)

    # Setup Rect
    for idx, clx in enumerate(classes):
        folder_path = os.path.join(TRAIN_PATH, clx)

        for raw_path in folder_path:
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detface = cascaderx.detectMultiScale(img, 1.2, 5)
            if len(detface) < 1: continue

            for x, y, w, h in detface:
                imgface = img[y:y+h, x:x+w]
                face_list.append(imgface)
                class_list.append(idx)

    # Train
    recognizerx.train(face_list, np.array(class_list))
    recognizerx.save(MODEL_PATH)
    print('Model Trained & Saved\n')

    # Init
    print('Initialize Testing...\n')
    total_images = 0
    total_correct = 0

    # Setup Rect & Test
    for true_name in os.listdir(TEST_PATH):
        if true_name not in classes: continue
        true_id = classes.index(true_name)
        folder_path = os.path.join(TEST_PATH, true_name)

        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detface = cascaderx.detectMultiScale(img, 1.2, 5)
            if len(detface) < 1: continue

            for x, y, w, h in detface:
                imgface = img[y:y+h, x:x+w]
                res_id, _ = recognizerx.predict(imgface)
                
                if res_id == true_id:
                    total_correct += 1
                total_images += 1

    # Accuracy
    if total_images > 0:
        acc = (total_images / total_correct) * 100
        print('Model Tested')
        print(f'Accuracy: {acc:.2f}%')
    else:
        print('No Face Image Found or Detected to Test')
    print('')

    return classes

def Predict ():
    