import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils import data
from torch.utils.data import Dataset


class maskImgDataset(Dataset):

    def __init__(self, path, image_size):
        super(maskImgDataset, self).__init__()
        self.image_size = image_size
        self.path = path
        self.list = os.listdir(path)
        """self.transform = DualCompose([
            Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            RandomCrop((self.image_size, self.image_size)),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])"""

    def get_mask_one(self, mask_path):
        if not os.path.exists(mask_path):
            mask_null = np.zeros([self.image_size, self.image_size, 3], np.uint8)
            mask_null = Image.fromarray(mask_null, 'RGB')
            return mask_null
        mask_one = Image.open(mask_path).convert("RGB")
        return mask_one

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")

        mask_skin = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_skin.png"
        mask_hair = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_hair.png"
        mask_l_brow = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_l_brow.png"
        mask_r_brow = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_r_brow.png"
        mask_l_eye = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_l_eye.png"
        mask_r_eye = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_r_eye.png"
        mask_nose = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_nose.png"
        mask_mouth = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_mouth.png"
        mask_l_lip = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_l_lip.png"
        mask_u_lip = os.path.dirname(self.path) + "/mask_" + str(self.image_size) + "/" + os.path.splitext(self.list[index])[0] + "_u_lip.png"

        mask_skin = self.get_mask_one(mask_skin)
        mask_hair = self.get_mask_one(mask_hair)
        mask_l_brow = self.get_mask_one(mask_l_brow)
        mask_r_brow = self.get_mask_one(mask_r_brow)
        mask_l_eye = self.get_mask_one(mask_l_eye)
        mask_r_eye = self.get_mask_one(mask_r_eye)
        mask_nose = self.get_mask_one(mask_nose)
        mask_mouth = self.get_mask_one(mask_mouth)
        mask_l_lip = self.get_mask_one(mask_l_lip)
        mask_u_lip = self.get_mask_one(mask_u_lip)

        '''# Resize
        image = functional.resize(image, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)])
        mask_skin = functional.resize(mask_skin, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_hair = functional.resize(mask_hair, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_l_brow = functional.resize(mask_l_brow, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_r_brow = functional.resize(mask_r_brow, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_l_eye = functional.resize(mask_l_eye, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_r_eye = functional.resize(mask_r_eye, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_nose = functional.resize(mask_nose, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_mouth = functional.resize(mask_mouth, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_l_lip = functional.resize(mask_l_lip, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)
        mask_u_lip = functional.resize(mask_u_lip, size=[int(self.image_size * 1.1), int(self.image_size * 1.1)], interpolation=functional.InterpolationMode.NEAREST)

        # RandomCrop
        crop_params = transforms.RandomCrop.get_params(image, (self.image_size, self.image_size))
        image = functional.crop(image, *crop_params)
        mask_skin = functional.crop(mask_skin, *crop_params)
        mask_hair = functional.crop(mask_hair, *crop_params)
        mask_l_brow = functional.crop(mask_l_brow, *crop_params)
        mask_r_brow = functional.crop(mask_r_brow, *crop_params)
        mask_l_eye = functional.crop(mask_l_eye, *crop_params)
        mask_r_eye = functional.crop(mask_r_eye, *crop_params)
        mask_nose = functional.crop(mask_nose, *crop_params)
        mask_mouth = functional.crop(mask_mouth, *crop_params)
        mask_l_lip = functional.crop(mask_l_lip, *crop_params)
        mask_u_lip = functional.crop(mask_u_lip, *crop_params)'''

        # ToTensor
        image = functional.to_tensor(image)
        mask_skin = torch.as_tensor(np.array(mask_skin).transpose(2,0,1), dtype=torch.int64)
        mask_hair = torch.as_tensor(np.array(mask_hair).transpose(2,0,1), dtype=torch.int64)
        mask_l_brow = torch.as_tensor(np.array(mask_l_brow).transpose(2,0,1), dtype=torch.int64)
        mask_r_brow = torch.as_tensor(np.array(mask_r_brow).transpose(2,0,1), dtype=torch.int64)
        mask_l_eye = torch.as_tensor(np.array(mask_l_eye).transpose(2,0,1), dtype=torch.int64)
        mask_r_eye = torch.as_tensor(np.array(mask_r_eye).transpose(2,0,1), dtype=torch.int64)
        mask_nose = torch.as_tensor(np.array(mask_nose).transpose(2,0,1), dtype=torch.int64)
        mask_mouth = torch.as_tensor(np.array(mask_mouth).transpose(2,0,1), dtype=torch.int64)
        mask_l_lip = torch.as_tensor(np.array(mask_l_lip).transpose(2,0,1), dtype=torch.int64)
        mask_u_lip = torch.as_tensor(np.array(mask_u_lip).transpose(2,0,1), dtype=torch.int64)

        # Normalize
        image = functional.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        mask_skin = mask_skin.clamp_(0, 1).int()
        '''mask_brow = (mask_l_brow + mask_r_brow).clamp_(0, 1)
        mask_eye = (mask_l_eye + mask_r_eye).clamp_(0, 1)
        mask_nose = mask_nose.clamp_(0, 1)
        mask_mouth = (mask_mouth + mask_l_lip + mask_u_lip).clamp_(0, 1)
        mask_hair = mask_hair.clamp_(0, 1)

        mask = torch.cat((mask_skin, mask_brow, mask_eye, mask_nose, mask_mouth, mask_hair), dim=0)'''

        mask_all = (mask_l_brow + mask_r_brow + mask_l_eye + mask_r_eye + mask_nose + mask_mouth + mask_l_lip + mask_u_lip).clamp_(0, 1).int()
        if torch.equal(mask_all, torch.zeros_like(mask_all)):
            mask_all = mask_skin
        mask = torch.cat((mask_skin, mask_all), dim=0)

        if image is not None:
            return image, mask

    def __len__(self):
        return len(self.list)


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = functional.resize(image, self.size)
        if target is not None:
            target = functional.resize(target, self.size, interpolation=functional.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            if target is not None:
                target = functional.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = transforms.RandomCrop.get_params(image, self.size)
        image = functional.crop(image, *crop_params)
        if target is not None:
            target = functional.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = functional.center_crop(image, self.size)
        if target is not None:
            target = functional.center_crop(target, self.size)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Pad:
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        image = functional.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = functional.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = functional.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
