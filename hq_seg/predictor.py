
import cv2
from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np


class Predictor(object):

    def __init__(self, model_file, x_range=(0, -1), y_step_size=3072, foreground_x_range=(280, 340)):
        sam = sam_model_registry["vit_b"](checkpoint=model_file)
        self.sam_predictor = SamPredictor(sam)
        if torch.cuda.is_available():
            self.sam_predictor.model.to('cuda:0')
            pass
        
        self.x_range = x_range
        self.y_step_size = y_step_size
        self.foreground_x_range = foreground_x_range
        pass

    def predict(self, img):
        # params: img: np.array, shape=(h, w, 3), order='BGR'

        masks = []
        for subimg in self.split_image(img, self.x_range, self.y_step_size):
            subimg = self.preprocess(subimg)
            x_range = self.foreground_x_range
            box = np.asarray([x_range[0], 0, x_range[1], subimg.shape[0]])
            # print(subimg.shape)
            # cv2.imshow(
            #     'subimg',
            #     cv2.rectangle(subimg, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 10))
            # cv2.waitKey(0)
            self.sam_predictor.set_image(subimg, image_format='BGR')
            mask, _, _ = self.sam_predictor.predict(box=box)
            mask = mask.astype('int8')
            mask = np.sum(mask, axis=0)
            mask = mask >= 2
            masks.append(mask)
            print(mask.shape)
            pass

        mask = np.concatenate(masks, axis=0)
        # print('last shape', mask.shape)
        return mask

    def preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1.0)
        mask = img_gray > 160
        img[mask] = 255
        # img[:, :x_range_except[0]] = 255
        # img[:, x_range_except[1]:] = 255
        return img

    def split_image(self, img, x_range, y_step_size):
        start = 0
        idx = 0
        
        while start < img.shape[0]:
            end = min(start+y_step_size, img.shape[0])
            
            if x_range[1] == -1:
                subimg = img[start: end, x_range[0]:]
            else:
                subimg = img[start: end, x_range[0]: x_range[1]]
                pass

            yield subimg
            start = end
            idx += 1
        pass

    pass