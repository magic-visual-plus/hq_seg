
import torch.utils.data
from torchvision.transforms import v2
import os
import cv2
from tqdm import tqdm


def get_default_transforms(image_size=1024):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, num_classes=None, image_size=1024):
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
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.filenames[idx]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.image_size, self.image_size))
        basename = os.path.splitext(self.filenames[idx])[0]
        mask_name = basename + '_mask.png'
        mask = cv2.imread(os.path.join(self.image_path, mask_name), cv2.IMREAD_GRAYSCALE)
        # resize mask
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask)
        mask = mask.long()

        img = self.transforms(img)

        return img, mask

    pass