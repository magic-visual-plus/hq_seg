import sys
import os
from hq_seg import predictor
import cv2
import numpy as np
from tqdm import tqdm

def draw_mask(img, mask):
    # mask = np.all(mask, axis=0)
    mask_img = np.zeros_like(img)
    # print(mask_img.shape)
    mask_img[:, :, 0] = 255
    mask_not = np.logical_not(mask)
    mask_img[mask_not] = img[mask_not]
    # cv2.copyTo(img, np.logical_not(mask), mask_img)
    
    img_ = cv2.addWeighted(img, 0.7, mask_img, 0.6, 0)
    
    return img_

if __name__ == '__main__':
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]

    seg_predictor = predictor.Predictor(model_path)

    filenames = os.listdir(input_path)
    filenames = [filename for filename in filenames if filename.endswith('.jpg')]

    for filename in tqdm(filenames):
        img =  cv2.imread(os.path.join(input_path, filename))
        mask = seg_predictor.predict(img, 0.7)

        img = draw_mask(img, mask)
        cv2.imwrite(os.path.join(output_path, filename), img)
        pass
    pass