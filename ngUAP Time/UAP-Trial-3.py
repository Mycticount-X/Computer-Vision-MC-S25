import os
import cv2
import math
import numpy as np

CASCADE_PATH = './haarcascade_frontalface_default.xml'
MODEL_PATH = './model_recognizer.xml'
TRAIN_PATH = './Dataset/train/'
TEST_PATH = './Dataset/test/'

def TrainTest ():
    # Train
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()
    
    face_list = []
    class_list = []
    classes = os.listdir(TRAIN_PATH)

    for idx, clx in enumerate(classes):
        folder_path = os.path.join(TRAIN_PATH, clx)
        
        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            detface = classifierx.detectMultiScale(img)
            if len(detface) < 1: continue

            for x, y, w, h in detface:
                imgface = img[y:y+h, x:x+h]
                face_list.append(imgface)
                class_list.append(idx)
    
    recognizerx.train(face_list, np.array(class_list))
    recognizerx.save(MODEL_PATH)

    # Test
    total_images = 0
    total_correct = 0

    for true_name in os.listdir(TEST_PATH):
        if true_name not in classes: continue
        true_id = classes.index(true_name)
        folder_path = os.path.join(TEST_PATH, true_name)
        
        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            detface = classifierx.detectMultiScale(img)
            if len(detface) < 1: continue

            for x, y, w, h in detface:
                imgface = img[y:y+h, x:x+w]
                res_id, _ = recognizerx.predict(imgface)
                
                if res_id == true_id:
                    total_correct += 1
                total_images += 1
        
    # Accuracy
    if total_images > 0:
        acc = (total_correct / total_images) * 100

        print('Model Trained')
        print(f'Accuracy: {acc:.2f}%')
    else:
        print('(!) No Face Images Found or Detected in Test Dataset')
    
    return classes

def Predict (classes):
    if not os.path.exists(MODEL_PATH):
        print('(!) Model not Found! Please Train the Model First!')
        return None
    
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()
    recognizerx.read(MODEL_PATH)

    # Input
    img_path = input('Please Input Target Image Path (Absolute Path): ')
    if not os.path.exists(img_path):
        print('(!) Invalid Path')
        return None

    img_ori = cv2.imread(img_path)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # Detect
    detface = classifierx.detectMultiScale(img)
    if len(detface) < 1:
        print('(!) No Face Detected')
        return None

    # Rect
    for x, y, w, h in detface:
        imgface = img[y:y+h, x:x+w]
        res_id, confidence = recognizerx.predict(imgface)
        confidence = math.floor(confidence * 100) / 100

        print(f'Detected Face: {classes[res_id]}')
        print(f'Confidence Level: {confidence}')

        cv2.rectangle(img_ori, (x,y), (x+w,y+h), (0, 255, 0), 2)
        text = f'{classes[res_id]}: {confidence}'
        cv2.putText(img_ori, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


    # Show
    cv2.imshow('Result', img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    classes = os.listdir(TRAIN_PATH) if os.path.exists(TRAIN_PATH) else None
    if classes is None:
        print('(!) Warning! Train Dataset Not Found!')
    
    while True:
        print('')
        print('PixelGate')
        print('1. Train and Test')
        print('2. Predict')
        print('3. Exit')
        cc = input('>> ')
        print('')

        if cc == '1':
            classes = TrainTest()
        elif cc == '2':
            Predict(classes)
        elif cc == '3':
            print('Alright, Have a Great Day!')
            break
        else:
            print('(!) Invalid Input')
    
