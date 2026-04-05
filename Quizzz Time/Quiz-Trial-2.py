import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def Preprocessing(img, blurtype='gaussian', ksize=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    if blurtype == 'gaussian':
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif blurtype == 'median':
        img = cv2.medianBlur(img, ksize)
    else:
        print('(!) Unknown Blur Detected!')
        print('    Using Gaussian Blur Instead')
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img

def Identify(img_path: str, dataset_path: str, algo='AKAZE', blurtype='gaussian', ksize=3):
    img_targ_ori = cv2.imread(img_path)
    if img_targ_ori is None:
        print('(!) Failed to Load Image')
        return None
    
    img_targ = Preprocessing(img_targ_ori, blurtype, ksize)

    # Desc Algo
    if algo == 'AKAZE':
        descriptor = cv2.AKAZE.create()
    elif algo == 'SIFT' or algo == 'SURF':
        descriptor = cv2.SIFT.create()
    else:
        print('(!) Unknown Descriptor Algorithm Detected!')
        print('    Using AKAZE Instead')
        descriptor = cv2.AKAZE.create()
    
    # KP & Desc Extract
    kp_targ, desc_targ = descriptor.detectAndCompute(img_targ, None)

    # Matcher
    desc_targ = np.float32(desc_targ)
    
    idx_params = dict(algorithm=1, trees=5)
    src_params = dict(checks=50)
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

        # KP & Desc
        kp_ref, desc_ref = descriptor.detectAndCompute(img_ref, None)
        desc_ref = np.float32(desc_ref)

        # Lowe's Ratio
        all_matches = matcher.knnMatch(desc_targ, desc_ref, k=2)
        good_matches = []

        for m, n in all_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Best Match
        if len(good_matches) > best_match:
            best_match = len(good_matches)
            best_data = {
                'name': filename,
                'image': img_ref_ori,
                'keypoint': kp_ref,
                'matches': good_matches
            }
    
    # Visualization
    if best_match > 0:
        print('PoVeKamon Terdeteksi!')
        print(f'[{algo}] | [{blurtype})]')
        print('')
        print('Best Match:')
        print(f'{best_data["name"]} ({len(best_data['matches'])} match(s))')
        print('')

        result = cv2.drawMatches(
            img_targ_ori, kp_targ,
            best_data['image'], best_data['keypoint'],
            best_data['matches'], None,
            (0, 0, 255), (255, 0, 255)
        )
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.imshow(result)
        plt.title(f'Target vs {best_data['name']} - {len(best_data['matches'])} match(s)')
        plt.axis(False)
        plt.show()
        
    else:
        print('Tidak ada PoVeKamon yang cocok dengan Gambar ini')



if __name__ == '__main__':
    Identify('./Images-Pokemon/Object.png', './Images-Pokemon/Data/', 'AKAZE', 'median', 5)

