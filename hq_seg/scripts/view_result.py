import sys
import cv2
import numpy as np

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
    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    img = cv2.imread(image_path)
    mask = np.load(mask_path)

    print(mask.shape)
    print(img.shape)
    img_ = draw_mask(img, mask)

    cv2.imshow('img', img_)
    cv2.waitKey(0)
    pass