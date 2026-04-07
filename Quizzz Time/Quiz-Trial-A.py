import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Preprocessing(img, blurtype='gaussian', ksize=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    
    if blurtype == 'gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blurtype == 'median':
        img = cv2.medianBlur(img, ksize)
    elif blurtype == 'avg':
        img = cv2.blur(img, (ksize, ksize))
    elif blurtype == 'bilateral':
        img = cv2.bilateralFilter(img, ksize, 75, 75)
    elif blurtype == 'filter2d':
        kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
        img = cv2.filter2D(img, -1, kernel)
    else:
        print("Unknown Filter Type")
        print("Using Gaussian Instead")
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    return img

def Identify(targ_path, dataset_path, algo='AKAZE', blurtype='gaussian', ksize=3):
    img_targ_ori = cv2.imread(targ_path)
    if img_targ_ori is None:
        print("Failed to Load Image")
        return None
    
    img_targ = Preprocessing(img_targ_ori, blurtype, ksize)

    # Desc Algo
    if algo == 'AKAZE':
        descriptor = cv2.AKAZE.create()
    elif algo == 'SIFT' or algo == 'SURF':
        descriptor = cv2.SIFT.create()
    elif algo == 'ORB':
        descriptor = cv2.ORB.create()
    else:
        print("Unknown Descriptor Algorithm")
        print("Using AKAZE Instead")
        descriptor = cv2.AKAZE.create()
    
    # KP & Desc Ext
    kp_targ, desc_targ = descriptor.detectAndCompute(img_targ, None)

    # Matcher
    if algo == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    else:
        desc_targ = np.float32(desc_targ)

        idx_params = dict(algorithm=1, trees=5)
        src_params = dict(checks=30)
        matcher = cv2.FlannBasedMatcher(idx_params, src_params)
    
    # Best Match
    best_match = 0
    best_data = {}

    for filename in os.listdir(dataset_path):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        ref_path = os.path.join(dataset_path, filename)
        img_ref_ori = cv2.imread(ref_path)
        img_ref = Preprocessing(img_ref_ori, blurtype, ksize)

        # KP & Desc Ext
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
            best_match = len(good_matches)
            best_data = {
                "name": filename,
                "image": img_ref_ori,
                "matches": good_matches,
                "keypoints": kp_ref
            }
    # Visualisasi
    if best_match > 0:
        print('Student Found')
        print('')
        print("Best Match:")
        print(f'{best_data["name"]} - {len(best_data["matches"])}')
        print('')

        result = cv2.drawMatches(
            img_targ_ori, kp_targ,
            best_data['image'], best_data['keypoints'],
            best_data['matches'], None,
            (0, 0, 255), (255, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10,5))
        plt.imshow(result)
        plt.title(f'Target vs {best_data["name"]}')
        plt.axis(False)
        plt.show()


    else:
        print('The Images is not match with any of Student Data')

if __name__ == '__main__':
    Identify('./Images-Biru/target/hina.png', './Images-Biru/source/', 'AKAZE')
    Identify('./Images-Biru/target/hina.png', './Images-Biru/source/', 'SIFT')
    Identify('./Images-Biru/target/hina.png', './Images-Biru/source/', 'ORB')