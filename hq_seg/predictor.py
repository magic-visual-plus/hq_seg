
import cv2
import torch
import numpy as np
from .models import image_segmenter
from . import datasets


class Predictor(object):

    def __init__(self, model_file, x_range=(0, -1), y_step_size=1024):
        model_data = torch.load(model_file, map_location='cpu')
        self.model = image_segmenter.ImageSegmenter(model_data['num_classes'])
        self.model.load_state_dict(model_data['state_dict'])
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            pass
        
        self.x_range = x_range
        self.y_step_size = y_step_size
        self.transforms = datasets.get_default_transforms(image_size=1024)
        pass

    def predict(self, img, threshold=0.1):
        # params: img: np.array, shape=(h, w, 3), order='BGR'
        # return: mask: np.array, shape=(h, w), dtype=np.uint8

        masks = []
        for subimg in self.split_image(img, self.x_range, self.y_step_size):
            original_h = subimg.shape[0]
            original_w = subimg.shape[1]

            subimg = self.preprocess(subimg)
            
            pixel_scores = self.model(subimg)
            pixel_scores = pixel_scores.squeeze(0)
            # resize to original size

            pred_mask = torch.argmax(pixel_scores, dim=0)

            pixel_proba = torch.nn.functional.softmax(pixel_scores, dim=0)
            mask_proba = pixel_proba > threshold
            mask_proba = mask_proba.any(dim=0)
            mask_proba = mask_proba.cpu().numpy()

            pred_mask = pred_mask.cpu().numpy()
            mask = np.zeros((subimg.shape[2], subimg.shape[3]), dtype=np.uint8)
            mask[pred_mask == 1] = 0
            mask[pred_mask == 2] = 1
            mask[np.logical_not(mask_proba)] = 0
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            masks.append(mask)
            pass

        mask = np.concatenate(masks, axis=0)
        return mask

    def preprocess(self, img):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.to('cuda:0')
            pass
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