# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self, data_name):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x
        # self.resize = transforms.Resize((375, 1242))
        self.data_name = data_name

    def __call__(self, sample, ):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        # image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': self.data_name}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class VKITTI2(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=False, split="train"):
        
        self.do_kb_crop = do_kb_crop
        self.transform = ToTensor(data_name="vkitti_2.0.3")

        # If train test split is not created, then create one.
        # Split is such that 10% of the frames from each scene are used for testing and validation.
        if not os.path.exists(os.path.join(data_dir_root, "vkitti2_train.txt")):
            import random
            import glob

            # image paths are of the form <data_dir_root>/vkitti_2.0.3_<rgb,depth>/<scene>/<variant>/frames/<rgb,depth>/Camera<0,1>/<rgb,depth>_{}.<jpg,png>
            self.image_files = glob.glob(os.path.join("/content/drive/MyDrive/magisterka/dane/vkitti2", "vkitti_2.0.3_rgb", "*", "*", "frames", "rgb", "*", '*.jpg'), recursive=True)
            self.depth_files = [r.replace("rgb", "depth").replace(".jpg", ".png") for r in self.image_files]
        
            scenes = set([f.split('/')[-6] for f in self.image_files])
            train_files = []
            valid_files = []
            test_files = []
            for scene in scenes:
                scene_files = [f for f in self.image_files if f.split('/')[-6] == scene]
                random.shuffle(scene_files)
                train_files.extend(scene_files[:int(len(scene_files) * 0.8)])
                valid_files.extend(scene_files[int(len(scene_files) * 0.8):int(len(scene_files) * 0.9)])
                test_files.extend(scene_files[int(len(scene_files) * 0.9):])
            with open(os.path.join(data_dir_root, "vkitti2_train.txt"), "w") as f:
                f.write("\n".join(train_files))
            with open(os.path.join(data_dir_root, "vkitti2_valid.txt"), "w") as f:
                f.write("\n".join(valid_files))
            with open(os.path.join(data_dir_root, "vkitti2_test.txt"), "w") as f:
                f.write("\n".join(test_files))

        
        with open(os.path.join(data_dir_root, "vkitti2_"+split+".txt"), "r") as f:
            self.image_files = f.read().splitlines()
        self.depth_files = [r.replace("rgb", "depth").replace(".jpg", ".png") for r in self.image_files]

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))

        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1

        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)

def get_vkitti2_loader(data_dir_root, batch_size=1, split="train",**kwargs):
    dataset = VKITTI2(data_dir_root, split=split)
    return DataLoader(dataset, batch_size, **kwargs)


class NYU2(Dataset):
    def __init__(self, data_dir_root):
        
        self.transform = ToTensor(data_name="nyu_2")

        # image paths are of the form <data_dir_root>/<rgb,depth>/{}.<jpg,png>
        if not os.path.exists(os.path.join(data_dir_root, "nyu_2_paths.txt")):
            import glob
            self.image_files = glob.glob(os.path.join("/content/drive/MyDrive/magisterka/dane/nyu_v2", "rgb", '*.jpg'), recursive=True)
            with open(os.path.join(data_dir_root, "nyu_2_paths.txt"), "w") as f:
                f.write("\n".join(self.image_files))
        else:
            with open(os.path.join(data_dir_root, "nyu_2_paths.txt"), "r") as f:
                self.image_files = f.read().splitlines()
        self.depth_files = [r.replace("rgb", "depth").replace(".jpg", ".png") for r in self.image_files]

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 1000.0  # mm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))
        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.

        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)

def get_nyu2_loader(data_dir_root, batch_size=1,**kwargs):
    dataset = NYU2(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


class KITTI(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=True, split="train"):
        
        self.do_kb_crop = do_kb_crop
        self.transform = ToTensor(data_name="kitti")

        # image paths are of the form <data_dir_root>\<raw_data,data_depth_annotated>\<scene>\<image_02\data, proj_depth\groundtruth\image_02>\{}.png
        if not os.path.exists(os.path.join(data_dir_root, "kitti_paths.txt")):
            import glob
            self.image_files = glob.glob(os.path.join("/content/drive/MyDrive/magisterka/dane/kitti", "raw_data", '*', 'image_02', 'data', '*.png'), recursive=True)
            with open(os.path.join(data_dir_root, "kitti_paths.txt"), "w") as f:
                f.write("\n".join(self.image_files))
        else:
            with open(os.path.join(data_dir_root, "kitti_paths.txt"), "r") as f:
                self.image_files = f.read().splitlines()
        self.depth_files = [r.replace("raw_data", "data_depth_annotated").replace("image_02\\data", "proj_depth\\groundtruth\\image_02") for r in self.image_files]
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))

        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1

        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)

def get_kitti_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = KITTI(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


class HYPERSIM(Dataset):
    def __init__(self, data_dir_root, split="train"):
        
        self.transform = ToTensor(data_name="hypersim")

        # If train test split is not created, then create one.
        # Split is such that 10% of the frames from each room are used for testing and validation.
        if not os.path.exists(os.path.join(data_dir_root, "hypersim_train.txt")):
            import random
            import glob
            
            # image paths are of the form <data_dir_root>/ai_*/images/scene_cam_00_<final_preview, geometry_hdf5>/frame.*.<color.jpg, depth_meters.png>
            self.image_files = glob.glob(os.path.join("/content/drive/MyDrive/magisterka/dane/hypersim", 'data', 'ai_*', 'images', 'scene_cam_00_final_preview', 'frame.*.color.jpg'), recursive=True)
            self.depth_files = [r.replace("final_preview", "geometry_hdf5").replace("color.jpg", "depth_meters.png") for r in self.image_files]
                    
            scenes = set([f.split('/')[-4] for f in self.image_files])
            train_files = []
            valid_files = []
            test_files = []
            for scene in scenes:
                scene_files = [f for f in self.image_files if f.split('/')[-4] == scene]
                random.shuffle(scene_files)
                train_files.extend(scene_files[:int(len(scene_files) * 0.8)])
                valid_files.extend(scene_files[int(len(scene_files) * 0.8):int(len(scene_files) * 0.9)])
                test_files.extend(scene_files[int(len(scene_files) * 0.9):])
            with open(os.path.join(data_dir_root, "hypersim_train.txt"), "w") as f:
                f.write("\n".join(train_files))
            with open(os.path.join(data_dir_root, "hypersim_valid.txt"), "w") as f:
                f.write("\n".join(valid_files))
            with open(os.path.join(data_dir_root, "hypersim_test.txt"), "w") as f:
                f.write("\n".join(test_files))


        with open(os.path.join(data_dir_root, "hypersim_"+split+".txt"), "r") as f:
            self.image_files = f.read().splitlines()
        self.depth_files = [r.replace("final_preview", "geometry_hdf5").replace("color.jpg", "depth_meters.png") for r in self.image_files]

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 1000.0  # mm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))
        
        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        
        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)

def get_hypersim_loader(data_dir_root, batch_size=1, split="train",**kwargs):
    dataset = HYPERSIM(data_dir_root, split=split)
    return DataLoader(dataset, batch_size, **kwargs)



if __name__ == "__main__":
    
    loader_vkitti2 = get_vkitti2_loader(data_dir_root="./data/vkitti2", split='train')
    print("Total files", len(loader_vkitti2.dataset))
    for i, sample in enumerate(loader_vkitti2):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        break
    

    loader_hypersim = get_hypersim_loader(data_dir_root="./data/hypersim", split='train')
    print("Total files", len(loader_hypersim.dataset))
    for i, sample in enumerate(loader_hypersim):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        break


    loader_nyu2 = get_nyu2_loader(data_dir_root="./data/nyu_v2")
    print("Total files", len(loader_nyu2.dataset))
    for i, sample in enumerate(loader_nyu2):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        break