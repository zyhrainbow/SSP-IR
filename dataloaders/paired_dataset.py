import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

# from .realesrgan import RealESRGAN_degradation

import pandas as pd
import sys
sys.path.append('/media/ps/ssd2/zyh/lv/emotion/SeeSR')
from models.model import get_text_index

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            use_n=False,
            text = "/gt_llm_caption"
            # null_dino_ratio=0.25,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        # self.null_dino_ratio = null_dino_ratio
        self.lr_list = []
        self.gt_list = []
        self.lq_caption_path_list = []
        self.use_n = use_n
        
       
        root_folders = root_folders.split(',')
       
        for root_folder in root_folders:
            lr_path = root_folder +'/sr_bicubic'
            lq_caption_path = root_folder + text
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.lq_caption_path_list += glob.glob(os.path.join(lq_caption_path, '*.txt'))
            # print(len(self.lr_list))
            

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.lq_caption_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        # ram_mean = [0.485, 0.456, 0.406]
        # ram_std = [0.229, 0.224, 0.225]
        # self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.275777]
        self.clip_normalize = transforms.Normalize(mean=clip_mean, std=clip_std)
        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return inputs.input_ids
    
    def tokenize_caption_n(self, caption=""):
        # inputs = self.tokenizer(
        #     caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        # )
        input_ids = get_text_index(self.tokenizer, caption)

        return input_ids

    def __getitem__(self, index):

        drop_dino_embedding = 0
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.lq_caption_path_list[index]
            
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()
        # if random.random() < self.null_dino_ratio:
        #     drop_dino_embedding = 1
       
        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        if self.use_n:
            example["input_ids"] = self.tokenize_caption_n(caption=tag).squeeze(0)
        else:
            example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)
        # example["caption"] = tag
        
        # example["drop_dino_embeds"] = drop_dino_embedding

        lq_img = lq_img.squeeze()

        # ram_values = F.interpolate(lq_img.unsqueeze(0), size=(224, 224), mode='bicubic')
        # ram_values = ram_values.clamp(0.0, 1.0)
        # example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))
        clip_values = F.interpolate(lq_img.unsqueeze(0), size=(224, 224), mode='bicubic')
        clip_values = clip_values.clamp(0.0, 1.0)
        example["clip_values"] = self.clip_normalize(clip_values.squeeze(0))
        
        return example

    def __len__(self):
        return len(self.gt_list)
    

class PairedPromptDataset(data.Dataset):
    def __init__(self, root_folders=None, tokenizer=None, null_text_ratio=0.5, text = "/llm_caption"):
        super(PairedPromptDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.lq_caption_path_list = []
        
       
        root_folders = root_folders.split(',')
       
        for root_folder in root_folders:
            lr_path = root_folder +'/sr_bicubic'
            lq_caption_path = root_folder + text
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.lq_caption_path_list += glob.glob(os.path.join(lq_caption_path, '*.txt'))
            # print(len(self.lr_list))
            

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.lq_caption_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.275777]
        self.clip_normalize = transforms.Normalize(mean=clip_mean, std=clip_std)
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.lq_caption_path_list[index]
            
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()
       
        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["caption"] = tag
        

        lq_img = lq_img.squeeze()
        clip_values = F.interpolate(lq_img.unsqueeze(0), size=(224, 224), mode='bicubic')
        clip_values = clip_values.clamp(0.0, 1.0)
        example["clip_values"] = self.clip_normalize(clip_values.squeeze(0))
        
        return example

    def __len__(self):
        return len(self.gt_list)

class PairedGTCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.25,
            null_dino_ratio=0.25,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedGTCaptionDataset, self).__init__()

        file = "/media/ps/ssd2/zyh/lv/emotion/SeeSR/preset/datasets/train_datasets/training_for_seesr/caption_train.csv"
        sep = "\t"
        img_key = "filepath"
        caption_key = "gt_caption"
        df = pd.read_csv(file, sep=sep)
        self.null_text_ratio = null_text_ratio
        self.null_dino_ratio = null_dino_ratio
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        
        print('Done loading data.')

        
        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
  
        lq_path = self.images[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)
        
        gt_path = lq_path.replace("sr_bicubic", "gt")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)

        if random.random() < self.null_text_ratio:
            caption = ''
        else:
            caption = self.captions[index]
        drop_dino_embedding = 0
        if random.random() < self.null_dino_ratio:
            drop_dino_embedding = 1

        
        

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=caption).squeeze(0)
        example["drop_dino_embeds"] = drop_dino_embedding

        lq_img = lq_img.squeeze()

        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(224, 224), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        return example

    def __len__(self):
        return len(self.images)
    
    
class PairedLQCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.25,
            null_dino_ratio=0.25,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            caption_type = 'lq_caption',
    ):
        super(PairedLQCaptionDataset, self).__init__()

        file = "/media/ps/ssd2/zyh/lv/emotion/SeeSR/preset/datasets/train_datasets/training_for_seesr/caption_train.csv"
        sep = "\t"
        img_key = "filepath"
        caption_key = "lq_caption"
        df = pd.read_csv(file, sep=sep)
        self.null_text_ratio = null_text_ratio
        self.null_dino_ratio = null_dino_ratio
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.gt_captions = df["gt_caption"].tolist()
        
        print('Done loading data.')

        
        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        drop_dino_embedding = 0
        lq_path = self.images[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)
        
        gt_path = lq_path.replace("sr_bicubic", "gt")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)

        if random.random() < self.null_text_ratio:
            caption = ''
            # gt_caption = ''
        else:
            caption = self.captions[index]
            # gt_caption = self.gt_captions[index]
        if random.random() < self.null_dino_ratio:
            drop_dino_embedding = 1

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=caption).squeeze(0)
        # example["gt_ids"] = self.tokenize_caption(caption=gt_caption).squeeze(0)
        example["drop_dino_embeds"] = drop_dino_embedding

        lq_img = lq_img.squeeze()

        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(224, 224), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        return example

    def __len__(self):
        return len(self.images)