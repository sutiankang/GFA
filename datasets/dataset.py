import os
import random

import numpy as np
import os.path as osp
from PIL import Image
from PIL import ImageEnhance

from torchvision import transforms
from torch.utils.data import Dataset


class UVOSDataset(Dataset):
    """
    image suffix: jpg
    flow suffix: jpg
    mask suffix: png
    """
    def __init__(self, cfg, is_train, datasets):

        self.train_support_dataset = ["YouTubeVOS-2018", "DAVIS-2016", "DAVIS-2017", "FBMS"]
        self.val_support_dataset = ["DAVIS-2016", "DAVIS-2017", "FBMS", "ViSal", "DAVSOD", "DAVSOD-Difficult",
                                    "DAVSOD-Normal", "SegTrack-V2", "Youtube-objects", "MCL"]

        img_size = cfg.img_size
        mean = cfg.mean
        std = cfg.std

        self.datasets = datasets
        self.data_dir = cfg.data_dir
        self.stride = cfg.stride
        self.is_train = is_train
        self.images, self.flows, self.masks = [], [], []

        # load dataset
        self.split_dataset(is_train=is_train)
        # transform
        size = self.get_size(img_size=img_size)
        self.image_transform = self.get_image_transform(size=size, mean=mean, std=std)
        self.flow_transform = self.get_flow_mask_transform(size=size)
        self.mask_transform = self.get_flow_mask_transform(size=size)

        assert len(self.images) == len(self.flows) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, flow, mask = self.images[idx], self.flows[idx], self.masks[idx]
        image = Image.open(image).convert("RGB")
        flow = Image.open(flow).convert("RGB")
        mask = Image.open(mask).convert("P")
        size = image.size  # original size
        image, flow, mask = self.data_augmentation(image, flow, mask)

        targets = {
            'image': image,
            'flow': flow,
            'mask': mask,
            'size': size,
            'path': self.images[idx]
        }

        return targets

    def load_dataset(self, dataset_name, is_train):

        if is_train:
            data_dir = osp.join(self.data_dir, dataset_name, "train")
        else:
            data_dir = osp.join(self.data_dir, dataset_name, "val")

        assert os.listdir(osp.join(data_dir, "images")) == os.listdir(osp.join(data_dir, "flows")) == \
               os.listdir(osp.join(data_dir, "labels")), \
               "video number or video name are different between images, flows and labels."

        videos = os.listdir(osp.join(data_dir, "images"))
        for video in videos:
            # image
            image_dir = osp.join(data_dir, "images", video)
            images_frames = sorted(os.listdir(image_dir))
            # flow
            flow_dir = osp.join(data_dir, "flows", video)
            flow_frames = sorted(os.listdir(flow_dir))
            # mask
            mask_dir = osp.join(data_dir, "labels", video)
            mask_frames = sorted(os.listdir(mask_dir))

            if len(mask_frames) < len(flow_frames):
                frames = mask_frames
            else:
                frames = flow_frames

            if is_train:
                if self.stride > 1 and dataset_name in ['YouTubeVOS-2018']:
                    frames = frames[::self.stride]

            for frame in frames:
                assert frame[:-4] + ".jpg" in images_frames
                assert frame[:-4] + ".jpg" in flow_frames
                assert frame[:-4] + ".png" in mask_frames

                self.images.append(osp.join(image_dir, frame[:-4] + ".jpg"))
                self.flows.append(osp.join(flow_dir, frame[:-4] + ".jpg"))
                self.masks.append(osp.join(mask_dir, frame[:-4]) + ".png")

    def split_dataset(self, is_train):
        if is_train:
            support_dataset = self.train_support_dataset
        else:
            support_dataset = self.val_support_dataset

        for dataset in self.datasets:
            if dataset in support_dataset:
                self.load_dataset(dataset_name=dataset, is_train=is_train)
            else:
                raise ValueError(f"Not support this dataset: {dataset}")

    def get_size(self, img_size):
        if isinstance(img_size, int):
            size = (img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
            size = (img_size[0], img_size[0])
        else:
            assert len(img_size) == 2, f"image size: {img_size} > 2 and is not a image"
            size = img_size
        return size

    def get_image_transform(self, size, mean, std):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform

    def get_flow_mask_transform(self, size):
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        return transform

    def data_augmentation(self, image, flow, mask):
        if self.is_train:
            image, flow, mask = self.random_crop(image, flow, mask, border=60)
            image, flow, mask = self.cv_random_flip(image, flow, mask)
            image,flow, mask = self.random_rotation(image, flow, mask)
            # image = self.random_peper(image)
            image = self.color_enhance(image)
        image = self.image_transform(image)
        flow = self.flow_transform(flow)
        mask = self.mask_transform(mask)

        return image, flow, mask

    def random_rotation(self, image, flow, mask):
        mode = Image.Resampling.BICUBIC
        if random.random() > 0.5:
            random_angle = np.random.randint(-10, 10)
            image = image.rotate(random_angle, mode)
            flow = flow.rotate(random_angle, mode)
            mask = mask.rotate(random_angle, mode)
        return image, flow, mask

    def cv_random_flip(self, image, flow, mask):
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flow = flow.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, flow, mask

    def color_enhance(self, image):
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return image

    def random_crop(self, image, flow, mask, border=30):
        image_width = mask.size[0]
        image_height = mask.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
            (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return image.crop(random_region), flow.crop(random_region), mask.crop(random_region)

    def random_peper(self, img):
        img = np.array(img)
        noiseNum = int(random.uniform(0, 0.1) * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[1] - 1)
            randY = random.randint(0, img.shape[0] - 1)
            if random.randint(0, 1) == 0:
                img[randY, randX] = 0
            else:
                img[randY, randX] = 255
        return Image.fromarray(img)