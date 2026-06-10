import cv2
import math
import os
import numpy as np

# Sesuaikan dengan struktur Latsol
TRAIN_PATH = "dataset/train/"
TEST_PATH = "dataset/test/"
MODEL_PATH = "face_recognizer_model.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def train_and_test():
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)
    recognizerx = cv2.face.LBPHFaceRecognizer_create()

    face_list = []
    class_list = []
    classes = os.listdir(TRAIN_PATH)

    # 1. Proses Training
    print("Training model...")
    for idx, clx in enumerate(classes):
        folder_path = os.path.join(TRAIN_PATH, clx)
        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            # Latsol meminta preprocess ke grayscale [cite: 129]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            detected_faces = classifierx.detectMultiScale(img, 1.2, 5)
            if len(detected_faces) < 1: continue

            for x, y, w, h in detected_faces:
                imgface = img[y:y+h, x:x+w]
                face_list.append(imgface)
                class_list.append(idx)
    
    recognizerx.train(face_list, np.array(class_list))
    
    # Menyimpan model sesuai kriteria soal 
    recognizerx.save(MODEL_PATH)

    # 2. Proses Testing & Hitung Akurasi
    total_images = 0
    correct_predictions = 0

    for true_name in os.listdir(TEST_PATH):
        if true_name not in classes: continue
        true_id = classes.index(true_name)

        folder_path = os.path.join(TEST_PATH, true_name)
        for raw_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, raw_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            detected_faces = classifierx.detectMultiScale(img, 1.2, 5)
            if len(detected_faces) < 1: continue

            for x, y, w, h in detected_faces:
                imgface = img[y:y+h, x:x+w]
                res_id, loss = recognizerx.predict(imgface)
                
                # Cek apakah tebakan sesuai dengan folder aslinya
                if res_id == true_id:
                    correct_predictions += 1
                total_images += 1
    
    # Cetak Akurasi akhir [cite: 132, 134]
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Average Accuracy: {accuracy:.2f}%")
    else:
        print("No test images found or detected.")
        
    return classes

def predict_custom_image(classes):
    # Validasi apakah model sudah ada 
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first.")
        return
    
    recognizerx = cv2.face.LBPHFaceRecognizer_create()
    recognizerx.read(MODEL_PATH)
    classifierx = cv2.CascadeClassifier(CASCADE_PATH)

    # Meminta input path dari user 
    img_path = input("Enter the path to the image for testing (absolute path): ")
    if not os.path.exists(img_path):
        print("Error: Image not found.")
        return

    img_ori = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    detected_faces = classifierx.detectMultiScale(img_gray, 1.2, 5)

    if len(detected_faces) < 1:
        print("No face detected.")
        return

    for x, y, w, h in detected_faces:
        imgface = img_gray[y:y+h, x:x+w]
        res_id, confidence = recognizerx.predict(imgface)
        confidence = math.floor(confidence * 100) / 100

        # Mencetak info di terminal sesuai contoh Latsol [cite: 143]
        print(f"Detected Subject: {classes[res_id]}")
        print(f"Confidence: {confidence}")

        # Menggambar kotak dan menaruh teks pada gambar 
        cv2.rectangle(img_ori, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{classes[res_id]} ({confidence})"
        cv2.putText(img_ori, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow('Predict Result', img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Memuat list class di awal agar opsi 2 bisa langsung jalan jika model sudah ada
    classes = os.listdir(TRAIN_PATH) if os.path.exists(TRAIN_PATH) else []

    while True:
        # Menu disamakan dengan Latsol [cite: 124]
        print('\n1. Train and test model.')
        print('2. Predict.')
        print('3. Exit')
        cc = input('Enter your choice: ')

        if cc == '1':
            classes = train_and_test()
        elif cc == '2':
            if not classes:
                print("Error: Dataset class not loaded.")
            else:
                predict_custom_image(classes)
        elif cc == '3':
            break
        else:
            print("Invalid input.")