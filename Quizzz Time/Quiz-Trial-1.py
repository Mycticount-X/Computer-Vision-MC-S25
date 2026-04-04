import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def Preprocessing(img, blurtype='gaussian', ksize=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    if blurtype == 'avg':
        img = cv2.blur(img, (ksize, ksize))
    elif blurtype == 'filter2d':
        kernel = np.ones((ksize, ksize), np.float32) / (ksize / ksize)
        img = cv2.filter2D(img, -1, kernel)
    elif blurtype == 'median':
        img = cv2.medianBlur(img, ksize)
    elif blurtype == 'gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blurtype == 'bilateral':
        img = cv2.bilateralFilter(img, ksize, 75, 75)
    else:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    return img

def Identify(img_path, dataset_path, algo='AKAZE', blurtype='gaussian', ksize=3):
    img_targ = cv2.imread(img_path)
    if img_targ is None:
        print('File tidak ditemukan')
        return None
    
    img_targ = Preprocessing(img_targ, blurtype, ksize)

    # Desc Algo
    if algo == 'AKAZE':
        descriptor = cv2.AKAZE.create()
    elif algo == 'SIFT':
        descriptor = cv2.SIFT.create()
    elif algo == 'ORB':
        descriptor = cv2.ORB.create()
    else:
        print('Algoritma tidak dikenali')
        print('Menggunakan Akaze sebagai pengganti')
        descriptor = cv2.AKAZE.create()
    
    # Keypoint & Desc Extraction
    kp_targ, desc_targ = descriptor.detectAndCompute(img_targ, None)

    if algo == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    else:
        desc_targ = np.float32(desc_targ)
        
        idx_params = dict(algorithm=1, trees=5)
        src_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(idx_params, src_params)
    
    # Best Match
    best_match = 0
    best_data = {}

    for filename in os.listdir(dataset_path):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        ref_path = os.path.join(dataset_path, filename)
        img_ref = cv2.imread(ref_path)
        img_ref = Preprocessing(img_ref, blurtype, ksize)

        # Keypoint & Desc Extraction
        kp_ref, desc_ref = descriptor.detectAndCompute(img_ref, None)

        if desc_ref is None or len(desc_ref) < 2:
            continue

        if algo != 'ORB':
            desc_ref = np.float32(desc_ref)
        
        # Lowe's Ratio Test
        matches = matcher.knnMatch(desc_targ, desc_ref, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Best Match
        if len(good_matches) > best_match:
            best_match = good_matches
            best_data = {
                'name': filename,
                'image': img_ref,
                'keypoints': kp_ref,
                'matches': good_matches
            }
    
    # Visualization
    if best_match > 0:
        print('PoveKamon Terdeteksi!!')
        print(f'[{algo} | {blurtype.upper()}]')
        print('')
        print('Best Match:')
        print(f'{best_data["name"]} ({best_data['matches']*100}%)')
        print('')

        result = cv2.drawMatches(
            img_targ, kp_targ,
            best_data['image'], best_data['keypoints'],
            best_data['matches'], None,
            matchColor=(255, 255, 0), singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.imshow(result)
        plt.title(f'Target vs {best_data["name"].capitalize()}')
        plt.axis('off')
        plt.show()

    else:
        print('Tidak ada PoveKamon yang cocok dengan Gambar ini')


if __name__ == '__main__':
    print('Version 1')
    Identify('./Images/Object.png', './Images/Data', 'AKAZE', 'median', 5)
    Identify('./Images/Object2.png', './Images/Data', 'AKAZE', 'median', 5)
    print('')

    print('Version 2')
    Identify('./Images/Object.png', './Images/Data', 'AKAZE', 'gaussian', 3)
    Identify('./Images/Object2.png', './Images/Data', 'AKAZE', 'gaussian', 3)