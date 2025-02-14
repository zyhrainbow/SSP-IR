import os
import random
import sys

import numpy as np
import pandas as pd

import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Mapping, Any

import random
import os
import cv2
import glob
import json
import math
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_for_seesr/', help='the dataset you want to tag.') # 
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 0/1/2/3/4 when using different GPU for parallel processing. It will save your time.') 
parser.add_argument("--all_gpu", type=int, default=1, help='if you set --start_gpu max to 5, please set it to 5') 
args = parser.parse_args()
mode = "train"
gt_path = os.path.join(args.root_path, 'gt')
lq_caption_path = os.path.join(args.root_path, 'lq_caption')
os.makedirs(lq_caption_path, exist_ok=True)
gt_caption_path = os.path.join(args.root_path, 'gt_caption')
os.makedirs(gt_caption_path, exist_ok=True)

gt_lists = glob.glob(os.path.join(gt_path, '*.png'))
print(f'There are {len(gt_lists)} imgs' )

start_num = args.start_gpu * len(gt_lists)//args.all_gpu
end_num = (args.start_gpu+1) * len(gt_lists)//args.all_gpu

print(f'===== process [{start_num}   {end_num}] =====')

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

# with torch.no_grad():
#     future_df = {"filepath":[], "gt_caption":[], "lq_caption":[]}
#     for gt_idx, gt_path in enumerate(gt_lists[start_num:end_num]):
#         print(f' ====== process {gt_idx} imgs... =====')
#         basename = os.path.basename(gt_path).split('.')[0]
#         lq_path = gt_path.replace('gt','sr_bicubic')
#         print(lq_path, gt_path)
#         gt = Image.open(gt_path).convert('RGB')
#         gt_caption = ci.generate_caption(gt)
#         lq = Image.open(lq_path).convert('RGB')
#         lq_caption = ci.generate_caption(lq)
#         print(f'The GT tag of {basename}.txt: {gt_caption}, {lq_caption}')

#         future_df["filepath"].append(lq_path)
#         future_df["gt_caption"].append(gt_caption)
#         future_df["lq_caption"].append(lq_caption)

#     pd.DataFrame.from_dict(future_df).to_csv(
#         os.path.join(args.root_path, f"caption_{mode}.csv"), index=False, sep="\t"
#     )
with torch.no_grad():
    for gt_idx, gt_path in enumerate(gt_lists[start_num:end_num]):
        print(f' ====== process {gt_idx} imgs... =====')
        basename = os.path.basename(gt_path).split('.')[0]
        lq_path = gt_path.replace('gt','sr_bicubic')
        gt = Image.open(gt_path).convert('RGB')
        gt_caption = ci.generate_caption(gt)
        lq = Image.open(lq_path).convert('RGB')
        lq_caption = ci.generate_caption(lq)
        gc_save_path = gt_caption_path + f'/{basename}.txt'
        f = open(f"{gc_save_path}", "w")
        f.write(gt_caption)
        f.close()
        print(f'The GT tag of {basename}.txt: {gt_caption}')
        lc_save_path = lq_caption_path + f'/{basename}.txt'
        f = open(f"{lc_save_path}", "w")
        f.write(lq_caption)
        f.close()
        print(f'The lq tag of {basename}.txt: {lq_caption}')

        