import cv2
import math
import os
import numpy as np

train_dir = 'images/train'
classes = os.listdir(train_dir)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# variabel global untuk menampung model yang abis dilatih
face_recog = None

def train_model():
    print('\n===Melakukan Training Model===')
    print('Loading.....')
   
    face_list = []
    class_list = []
   
    for index, person_name in enumerate(classes):
        class_path = train_dir + "/" + person_name
        for image_path in os.listdir(class_path):
            full_image_path = class_path + "/" + image_path
           
            img_gray = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
           
            if len(detected_faces) < 1:
                continue
           
            for rect in detected_faces:
                x,y,w,h = rect
                # [Start:Stop]
                # [Baris: Column]
                face_img = img_gray[y:y+h, x:x+w]
                face_list.append(face_img)
                class_list.append(index)
   
    if len(face_list) == 0:
        print('Tidak ada wajah')
        return None
   
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(face_list, np.array(class_list))
    return model

def test_model(model):
    print('\nTesting Model....')
   
    if model is None:
        print('Model belum dilatih')
        return
   
    test_dir = 'images/test'
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print('Folder Error')
        return
   
    for image_path in os.listdir(test_dir):
        full_image_path = test_dir + "/" + image_path
       
        img_bgr = cv2.imread(full_image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
       
        detected_faces = face_cascade.detectMultiScale(img_gray, 1.2, 5)
       
        if len(detected_faces) < 1:
            continue
       
        for rect in detected_faces:
            x,y,w,h = rect
            face_img = img_gray[y:y+h, x:x+w]
           
            res, confidence = model.predict(face_img)
            confidence = math.floor(100 * confidence) / 100
           
            cv2.rectangle(img_bgr, (x,y), (x+w,y+h), (0,255,0), 1)
            text = classes[res] + " " + str(confidence) + "%"
            cv2.putText(img_bgr, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 1)
           
            cv2.imshow("Result", img_bgr)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
   
while True:
    print("\n=======================\n")
    print("        Face Recog       ")
    print("==========================")
    print("1. Train Model")
    print("2. Test Model")
    print("3. Exit")
    print("===========================")
    choice = input(">>>")
   
    if choice == "1":
        face_recog = train_model()
    elif choice == "2":
        test_model(face_recog)
    elif choice == "3":
        print("Thank You....")
        break
    else:
        print("Input Not Valid")