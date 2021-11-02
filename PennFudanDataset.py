"""
import os
from typing import Any
import numpy as np
import torch as t
from torch.utils.data import Dataset
from PIL import Image

class PennFudanDataset(Dataset):
    def __init__(self, root="data/PennFudanPed", transforms=None) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, index) -> Any:
        img_path = os.path.join(self.root, "PNGImages", self.imgs[index])
        msk_path = os.path.join(self.root, "PedMasks", self.masks[index])
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path)
        msk = np.array(msk)
        obj_ids = np.unique(msk)
        # remove zero index id (we assume that it is a background)
        obj_ids = obj_ids[1:]
        masks = msk == obj_ids[:, None, None]
        # get bounding box
        num_obj = len(obj_ids)
        boxes = []
        for i in range(num_obj):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            ymin = np.min(pos[0])
            xmax = np.max(pos[1])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = t.as_tensor(boxes, dtype=t.float32)
        lbl = t.ones((num_obj,), dtype=t.int64) # two object, treated as one class pedestrian
        img_id = t.tensor([index])
        area = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        iscrowd = t.zeros((num_obj,), dtype=t.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = lbl
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
"""
import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        #print(self.transforms)

        if self.transforms is not None:
            img = self.transforms(img)
            #img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)