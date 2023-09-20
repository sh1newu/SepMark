import os
import torch
import torch.nn as nn
import argparse
from torch.backends import cudnn
from PIL import Image
import random, string
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from network.noise_layers.stargan.model import Generator


class StarGAN(nn.Module):

    def __init__(self, c_trg=3, image_size=256, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(StarGAN, self).__init__()
        self.c_trg = c_trg
        self.image_size = image_size
        self.temp = temp
        self.target = target
        self.attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.G = Generator(64, 5, 6)
        if not os.path.exists(temp): os.mkdir(temp)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_temp(self):
        return self.temp + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png"  # png or jpg?

    def get_target(self):
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_temp()
            while os.path.exists(file):
                file = self.get_temp()

            try:
                self.test(self.get_target(), im, mask[i], file)
                fake = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except:
                print("Error in stargan")
                noised_image[i] = image[i]

        return noised_image

    def test(self, face_image_path, body_image, label, output_path):
        regular_image_transform = []
        regular_image_transform.append(transforms.ToTensor())
        regular_image_transform.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        regular_image_transform = transforms.Compose(regular_image_transform)

        G_path = os.path.join('network/noise_layers/stargan', str(self.image_size),'200000-G.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{0}'))
        self.G = self.G.cuda(0)

        with torch.no_grad():
            x_real = regular_image_transform(body_image).unsqueeze(0).cuda()
            c_org = label

            # Prepare input images and target domain labels.
            x_real = x_real.to("cuda")
            c_trg_list = self.create_labels(c_org, 5, selected_attrs=self.attrs)

            if self.c_trg is None:
                self.c_trg = random.randint(0, 4)
            # Translate images.
            x_fake = self.G(x_real, c_trg_list[self.c_trg].unsqueeze(0))

            # Save the translated images.
            save_image(self.denorm(x_fake), os.path.join(output_path))

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[j] = 0
                else:
                    c_trg[i] = (c_trg[i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to("cuda"))
        return c_trg_list