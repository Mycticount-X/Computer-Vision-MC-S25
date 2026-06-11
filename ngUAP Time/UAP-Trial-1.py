import cv2
import math
import os
import numpy as np

TRAIN_PATH = './Dataset/train/'
TEST_PATH = './Dataset/test/'
MODEL_PATH = './model_recognizer.xml'
CASCADE_PATH = './haarcascade_frontalface_default.xml'

def TrainTest ():
    # Init
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()

    face_list = []
    class_list = []
    classes = os.listdir(TRAIN_PATH)

    # Setup Rect
    for idx, clx in enumerate(os.listdir(TRAIN_PATH)):
        folder_path = os.path.join(TRAIN_PATH, clx)
        
        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detface = classifierx.detectMultiScale(img, 1.2, 5)
            if (len(detface) < 1): continue

            for x, y, w, h in detface:
                imgface = img[y:y+h,x:x+w]
                face_list.append(imgface)
                class_list.append(idx)

    # Train
    recognizerx.train(face_list, np.array(class_list))
    recognizerx.save(MODEL_PATH)

    # Init 
    total_images = 0
    total_correct = 0

    # Setup Rect
    for true_name in os.listdir(TEST_PATH):
        if true_name not in classes: continue
        true_id = classes.index(true_name)
        folder_path = os.path.join(TEST_PATH, true_name)
        
        for raw_img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_img)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detface = classifierx.detectMultiScale(img, 1.2, 5)
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
        print(f'Accuracy: {acc:.2f}%')
    else:
        print('No Face Image Found or Detected')
    
    return classes

def Predict (classes):
    # Init & Check
    if not os.path.exists(MODEL_PATH):
        print('(!) Model not Found! Please Train the Model First')
        return None
    
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()
    recognizerx.read(MODEL_PATH)

    # Input
    img_path = input('Please Input Your Image Path (Absolute Path): ')
    if not os.path.exists(img_path):
        print('(!) Image Not Found')
        return None

    img_ori = cv2.imread(img_path)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # Detect
    detface = classifierx.detectMultiScale(img, 1.2, 5)

    if len(detface) < 1:
        print('(!) No Face Detected')
        return None
    
    # Draw Rect
    for x, y, w, h in detface:
        imgface = img[y:y+h, x:x+w]
        res_id, confidence = recognizerx.predict(imgface)
        confidence = math.floor(confidence * 100) / 100

        print(f'Detected Subject: {classes[res_id]}')
        print(f'Confidence Level: {confidence}')

        cv2.rectangle(img_ori, (x,y), (x+w, y+h), (0, 255, 0), 2)
        text = f'{classes[res_id]} : {confidence}'
        cv2.putText(img_ori, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    
    # Show
    cv2.imshow("Result", img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    classes = os.listdir(TRAIN_PATH) if os.path.exists(TRAIN_PATH) else None
    if classes is None:
        print('(!) Warning: Dataset Classes not Found!')
    
    while True:
        print('')
        print('1. Train and Test')
        print('2. Predict')
        print('3. Exit')
        cc = input('>> ')
        print('')

        if cc == '1':
            classes = TrainTest()
        elif cc == '2':
            if classes is None:
                print('Cannot do the Prediction due to No Dataset Classes')
            else:
                Predict(classes)
        elif cc == '3':
            print('Bye')
            break
        else:
            print('(!) Invalid Input')
    