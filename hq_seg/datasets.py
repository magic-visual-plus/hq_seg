
import torch.utils.data
from torchvision.transforms import v2
import os
import cv2
from tqdm import tqdm
import random
import numpy as np


class RandomCrop:
    def __init__(self, ):
        pass

    def __call__(self, img, mask):
        # if random.random() < 0.1:
        #     pass
        # elif random.random() < 0.5:
        #     shift_left = random.randint(1, 100)
        #     img = np.concatenate([img[:, shift_left:], img[:, :shift_left]], axis=1)
        #     mask = np.concatenate([mask[:, shift_left:], mask[:, :shift_left]], axis=1)
        #     pass
        # else:
        #     shift_right = random.randint(1, 100)
        #     img = np.concatenate([img[:, -shift_right:], img[:, :-shift_right]], axis=1)
        #     mask = np.concatenate([mask[:, -shift_right:], mask[:, :-shift_right]], axis=1)
        #     pass
        if random.random() < 0.2:
            pass
        else:
            shift_left = random.randint(0, 100)
            shift_right = random.randint(0, 100)

            h = img.shape[0]
            w = img.shape[1]

            img = img[:, shift_left:w-shift_right]
            mask = mask[:, shift_left:w-shift_right]
            if h > 900:
                shift_top = random.randint(0, 300)
                shift_bottom = random.randint(0, 300)
                img = img[shift_top:h-shift_bottom, :]
                mask = mask[shift_top:h-shift_bottom, :]
                pass
            
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
            pass

        return img, mask



def get_default_transforms(image_size=1024):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_random_transforms(image_size=1024):
    return v2.Compose([
        RandomCrop(),
    ])

class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, num_classes=None, image_size=1024, random_transform=False):
        self.image_path = image_path
        self.filenames = []
        for filename in os.listdir(image_path):
            if not filename.endswith('_mask.png'):
                # it is not a mask

                # check if there is a mask
                mask_name = os.path.splitext(filename)[0] + '_mask.png'
                if os.path.exists(os.path.join(image_path, mask_name)):
                    # this file is an image and has mask
                    self.filenames.append(filename)
                    pass
                pass
            pass
        self.image_size = image_size
        
        if num_classes is not None:
            self.num_classes = num_classes
            pass
        else:
            # calculate num_classes from datasets
            self.num_classes = None
            for filename in tqdm(self.filenames):
                basename = os.path.splitext(filename)[0]
                mask_name = basename + '_mask.png'
                mask = cv2.imread(os.path.join(image_path, mask_name), cv2.IMREAD_GRAYSCALE)
                # print(mask.max())
                if self.num_classes is None:
                    self.num_classes = mask.max() + 1
                else:
                    self.num_classes = max(self.num_classes, mask.max() + 1)
                    pass
                pass
            pass

        self.transforms = get_default_transforms(image_size)
        self.transforms_rand = get_random_transforms(image_size)
        self.random_transform = random_transform
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.filenames[idx]), cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (self.image_size, self.image_size))
        basename = os.path.splitext(self.filenames[idx])[0]
        mask_name = basename + '_mask.png'
        mask = cv2.imread(os.path.join(self.image_path, mask_name), cv2.IMREAD_GRAYSCALE)
        # resize mask
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        
        if self.random_transform:
            img, mask = self.transforms_rand(img, mask)
            pass
        #print(mask.shape)
        img = self.transforms(img)
        mask = cv2.resize(mask, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST_EXACT)
        mask = torch.from_numpy(mask)
        mask = mask.long()

        return img, mask

    pass


class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.filenames = os.listdir(path)
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        arrays = np.load(os.path.join(self.path, self.filenames[idx]))


        return arrays['patch'], arrays['patch_mask'], arrays['x']

    pass