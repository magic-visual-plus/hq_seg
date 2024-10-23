import unittest
from hq_seg.predictor import Predictor
import os
import numpy as np
import cv2


class TestSegment(unittest.TestCase):
    def test_segment(self):
        img_path = os.path.join('data', 'test.bmp')

        predictor = Predictor(os.path.join('data', 'sam_vit_h_4b8939.pth'))
        img = cv2.imread(img_path)
        mask = predictor.predict(img)

        np.save(os.path.join('data', 'mask.npy'), mask)
        self.assertTrue(mask is not None)
        pass
    pass


if __name__ == '__main__':
    unittest.main()
    pass