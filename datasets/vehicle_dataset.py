import glob
import os
from typing import List

import torchvision
from PIL import Image
from tqdm import tqdm
# from StableDiffusion.utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import cv2
import torch
import numpy as np
import random

class VehicleDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """

    def __init__(self, split, im_path, im_channels,im_size = 512, im_ext = 'jpg', shuffle=False, subset = None,
                 use_latents=False, return_hint = False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.shuffle = shuffle
        self.subset = subset
        self.return_hints = return_hint
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        # self.idx_to_cls_map = {}
        # self.cls_to_idx_map = {}

        # if 'image' in self.condition_types:
        #     self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
        #     self.mask_h = condition_config['image_condition_config']['image_condition_h']
        #     self.mask_w = condition_config['image_condition_config']['image_condition_w']

        # self.images, self.texts, self.masks = self.load_images(im_path)
        self.images, self.labels = self.load_images(im_path)

        # Whether to load images and call vae or to load latents
        # if use_latents and latent_path is not None:
        #     latent_maps = load_latents(latent_path)
        #     if len(latent_maps) == len(self.images):
        #         self.use_latents = True
        #         self.latent_maps = latent_maps
        #         print('Found {} latents'.format(len(self.latent_maps)))
        #     else:
        #         print('Latents not found')
        print('Found {} images , {} text, '.format(len(self.images), len(self.labels)))

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []

        for d_name in tqdm(os.listdir(im_path)):
            # print(self.subset)
            # if self.shuffle:
            #     random.shuffle(d_name)
            # if self.subset is not None:
            #     d_name = d_name[:self.subset]

            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpeg')))


            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_types:
                    labels.append(d_name)
        print('Found {} images for split {}'.format(len(ims), self.split))
        print('Found {} captions'.format(len(labels)))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'class' in self.condition_types:
            cond_inputs['class'] = self.labels[index]
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if self.return_hints:
                canny_image = Image.open(self.images[index])
                canny_image = np.array(canny_image)
                canny_image = cv2.Canny(canny_image, 100, 200)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
                return latent, canny_image_tensor
            else:
                return latent

        else:
            # Load and preprocess the image
            im = Image.open(self.images[index])

            # Convert to RGB if single channel
            if im.mode != 'RGB':
                im = im.convert('RGB')

            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
                # Add channel dimension check
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Updated for 3 channels
            ])(im)
            im.close()
            # Check for NaN or Inf values
            if torch.isnan(im_tensor).any() or torch.isinf(im_tensor).any():
                # Skip this image and return a random valid image instead
                return self.__getitem__(random.randint(0, len(self) - 1))
                im_tensor = (2 * im_tensor) - 1

            # Check for NaN or Inf values
            assert not torch.isnan(im_tensor).any(), "Image tensor contains NaN values"
            assert not torch.isinf(im_tensor).any(), "Image tensor contains Inf values"

            # Normalize the image tensor
            # im_tensor = (2 * im_tensor) - 1

            if self.return_hints:
                # Load the image and resize to 512x512
                canny_image = Image.open(self.images[index])
                canny_image = canny_image.resize((self.im_size, self.im_size))  # Resize here
                canny_image = np.array(canny_image)

                # Check if the image is valid
                assert canny_image is not None, "Input image is None"
                assert canny_image.size > 0, "Input image is empty"

                # Convert to grayscale if the image is RGB (3 channels)
                if len(canny_image.shape) == 3:
                    canny_image = cv2.cvtColor(canny_image, cv2.COLOR_RGB2GRAY)

                # Apply Canny edge detection
                canny_image = cv2.Canny(canny_image, 100, 200)
                # Check for NaN or Inf values in the Canny output
                if np.isnan(canny_image).any() or np.isinf(canny_image).any():
                    # Skip this image and return a random valid image instead
                    return self.__getitem__(random.randint(0, len(self) - 1))

                # Check for NaN or Inf values in the Canny output
                assert not np.isnan(canny_image).any(), "Canny output contains NaN values"
                assert not np.isinf(canny_image).any(), "Canny output contains Inf values"

                # Convert to 3-channel "RGB" format (optional, but matches common tensor shapes)
                canny_image = canny_image[:, :, None]  # Add channel axis (now 512x512x1)
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)  # 512x512x3

                # Convert to PyTorch tensor
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)  # Shape: (3, 512, 512)

                return im_tensor, canny_image_tensor, cond_inputs
            else:
                return im_tensor


if __name__ == "__main__":
    datasets = VehicleDataset('val',
                              im_path='D:/new_folder/JOSEPH_FILe/python/AdvancedAI/AI/Diffusion/data/data/vehicle/test',
                              im_channels=3,
                              im_size=512, shuffle=True, subset=300)
    print(datasets[4])
