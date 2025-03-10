
import cv2
import torch
import numpy as np
from .models import image_segmenter
from . import datasets
from typing import Tuple


class Predictor(object):

    def __init__(self, model_file, step_size: Tuple[int, int] = (-1, -1)):
        model_data = torch.load(model_file, map_location='cpu')
        self.model = image_segmenter.ImageSegmenter(model_data['num_classes'])
        # self.model = image_segmenter.ImageSegmenter2(
        #     model_data['num_classes'],
        #     encoder_size=32, hidden_size=512, decoder_size=32,
        #     kernel_size=8, stride=8,
        #     num_layers_encoder=3, num_layers_hidden=6, num_layers_decoder=3
        # )
        self.model.load_state_dict(model_data['state_dict'])
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            pass
        
        self.step_size = step_size
        self.transforms = datasets.get_default_transforms(image_size=256)
        pass

    def predict(self, img, threshold=0.1):
        # params: img: np.array, shape=(h, w, 3), order='BGR'
        # return: mask: np.array, shape=(h, w), dtype=np.uint8

        masks = []
        subimgs = []
        x_step, y_step = self.step_size

        if x_step == -1:
            x_step = img.shape[1]
            pass

        if y_step == -1:
            y_step = img.shape[0]
            pass

        for y in range(0, img.shape[0], y_step):
            for x in range(0, img.shape[1], x_step):
                subimg = img[y:y+y_step, x:x+x_step]

                subimg = self.preprocess(subimg)
                subimgs.append(subimg)
                pass
            pass

        subimgs = torch.cat(subimgs, dim=0)
        with torch.no_grad():
            pixel_scores = self.model(subimgs)
            pass

        idx = 0
        y_masks = []
        for y in range(0, img.shape[0], y_step):
            x_masks = []
            for x in range(0, img.shape[1], x_step):
                pixel_score = pixel_scores[idx]
                idx += 1

                original_h = min(y_step, img.shape[0] - y)
                original_w = min(x_step, img.shape[1] - x)

                pixel_score = pixel_score.unsqueeze(0).unsqueeze(0)
                pixel_score = torch.nn.functional.interpolate(pixel_score, (original_h, original_w), mode='bilinear', align_corners=False)
                pixel_score = pixel_score.squeeze(0).squeeze(0)
                # resize to original size

                pred_mask = (pixel_score > 0).float()
                pixel_proba = torch.nn.functional.sigmoid(pixel_score)
                mask_proba = pixel_proba > threshold
                mask_proba = mask_proba.cpu().numpy()

                pred_mask = pred_mask.cpu().numpy()
                mask = pred_mask.astype(np.uint8)
                mask[np.logical_not(mask_proba)] = 0

                mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR_EXACT)
                x_masks.append(mask)
                pass
            y_mask = np.concatenate(x_masks, axis=1)
            y_masks.append(y_mask)
            pass

        mask = np.concatenate(y_masks, axis=0)
        return mask

    def preprocess(self, img):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.to('cuda:0')
            pass
        return img

    pass