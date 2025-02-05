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
    # mask_img[mask>0] = 255
    mask_not = np.logical_not(mask)
    mask_img[mask_not] = img[mask_not]
    # cv2.copyTo(img, np.logical_not(mask), mask_img)
    
    img_ = cv2.addWeighted(img, 0.7, mask_img, 0.6, 0)
    # img_ = mask_img
    
    return img_

def draw_width_min_max(mask, img):
    width = mask.shape[1]
    r = np.arange(width)
    rimg = np.tile(r, (mask.shape[0], 1))
    rmask_max = mask * rimg
    rmask_min = rimg + (1 - mask) * (width + 1)

    xmin = np.min(rmask_min, axis=1)
    xmax = np.max(rmask_max, axis=1)
    xwidth = xmax - xmin

    min_width_y = np.argmin(xwidth)
    min_width_x_start = xmin[min_width_y]
    min_width_x_end = xmax[min_width_y]

    max_width_y = np.argmax(xwidth)
    max_width_x_start = xmin[max_width_y]
    max_width_x_end = xmax[max_width_y]

    img = cv2.line(img, (min_width_x_start, min_width_y), (min_width_x_end, min_width_y), (0, 255, 0), 3)
    img = cv2.line(img, (max_width_x_start, max_width_y), (max_width_x_end, max_width_y), (0, 0, 255), 3)
    return img

if __name__ == '__main__':
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]

    seg_predictor = predictor.Predictor(model_path)

    filenames = os.listdir(input_path)
    filenames = [filename for filename in filenames if filename.endswith('.jpg')]

    for filename in tqdm(filenames):
        img =  cv2.imread(os.path.join(input_path, filename))
        mask = seg_predictor.predict(img)
        img = draw_mask(img, mask)
        img = draw_width_min_max(mask, img)
        cv2.imwrite(os.path.join(output_path, filename), img)
        pass
    pass