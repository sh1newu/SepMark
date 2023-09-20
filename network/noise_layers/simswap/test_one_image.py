import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from network.noise_layers.simswap.models.models import create_model
from network.noise_layers.simswap.test_options import TestOptions
import os
import torch.nn as nn
import random, string
import warnings
warnings.filterwarnings("ignore")


class SimSwap(nn.Module):
    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(SimSwap, self).__init__()
        self.temp = temp
        self.target = target
        if not os.path.exists(temp): os.mkdir(temp)

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        opt = TestOptions().parse()

        # torch.nn.Module.dump_patches = True
        self.model = create_model(opt)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def get_temp(self):
        return self.temp + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png" # png or jpg?

    def get_target(self):
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def forward(self, image_cover_mask):
        self.model.eval()
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            #########################################################
            pic_a = self.get_target()
            img_a = Image.open(pic_a).convert('RGB')
            img_a = self.transformer_Arcface(img_a)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            img_b = self.transformer(im)
            img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()
            img_att = img_att.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = latend_id.detach().to('cpu')
            latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
            latend_id = latend_id.to('cuda')

            ############## Forward Pass ######################
            img_fake = self.model(img_id, img_att, latend_id, latend_id, True)

            for i in range(img_id.shape[0]):
                if i == 0:
                    row1 = img_id[i]
                    row2 = img_att[i]
                    row3 = img_fake[i]
                else:
                    row1 = torch.cat([row1, img_id[i]], dim=2)
                    row2 = torch.cat([row2, img_att[i]], dim=2)
                    row3 = torch.cat([row3, img_fake[i]], dim=2)

            # full = torch.cat([row1, row2, row3], dim=1).detach()
            full = row3.detach()
            #full = full.permute(1, 2, 0)
            #output = full.to('cpu')
            #output = np.array(output)
            output = (full.clamp(0, 1).permute(1, 2, 0) * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            output = output[..., ::-1]

            #output = output * 255
            #########################################################

            file = self.get_temp()
            while os.path.exists(file):
                file = self.get_temp()
            cv2.imwrite(file, output)
            fake = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)

        return noised_image
