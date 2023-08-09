import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class ImgDataset(Dataset):

    def __init__(self, path, image_size):
        super(ImgDataset, self).__init__()
        self.image_size = image_size
        self.path = path
        self.list = os.listdir(path)
        self.transform = transforms.Compose([
            #transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            #transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
        image = self.transform(image)
        if image is not None:
            return image, torch.zeros_like(image)

    def __len__(self):
        return len(self.list)
