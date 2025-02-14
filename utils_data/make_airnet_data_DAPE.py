'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import cv2
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import argparse


from torchvision.transforms import Normalize, Compose
from torchvision import transforms
import glob
from PIL import Image
import random
import torch.nn.functional as F




root_path = "/media/ps/ssd2/zyh/dataset/dape_air/train"

gt_path = os.path.join(root_path, 'gt')
lq_path = os.path.join(root_path, 'lq')

os.makedirs(gt_path, exist_ok=True)
os.makedirs(lq_path, exist_ok=True)


noisy_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/DIV2K/DIV2K_train_noisy", "/media/ps/ssd2/zyh/dataset/airnet/Flickr2K/Flickr2K_noisy"]
jpeg_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/DIV2K/DIV2K_train_jpeg", "/media/ps/ssd2/zyh/dataset/airnet/Flickr2K/Flickr2K_jpeg"]
rain_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/universal/train/rainy/RainTrainH"]
rain2_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/Train/Derain/rainy"]
low_light_dirs = ["/media/ps/ssd2/zyh/lv/dataset/LOLdataset/our485/low"]
haze_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/universal/train/hazy/train/LQ"]
snow_dirs = ["/media/ps/ssd2/zyh/dataset/airnet/universal/train/snowy/lq"]
blur_dirs =["/media/ps/ssd2/zyh/dataset/airnet/universal/train/gopro/train"]

ratio = [1, 1, 1, 8, 1, 1]

fix_size = 512
exts = ['*.jpg', '*.png']


noisy_img_list = []
jpeg_img_list = []
rain_img_list = []
rain2_img_list = []
low_light_img_list = []
haze_img_list = []
snow_img_list = []
blur_img_list = []

for noisy_dir in noisy_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(noisy_dir, ext))
        noisy_img_list += image_list

print(f"noisy number: {len(noisy_img_list)}")
        
for jpeg_dir in jpeg_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(jpeg_dir, ext))
        jpeg_img_list += image_list
        
print(f"jpeg number: {len(jpeg_img_list)}")      
        
for rain_dir in rain_dirs:
    
    
    image_list = glob.glob(os.path.join(rain_dir, "rain-*.png"))
    rain_img_list += image_list

        
print(f"rain number: {len(rain_img_list)}") 

for rain2_dir in rain2_dirs:
    
    image_list = glob.glob(os.path.join(rain2_dir, "rain-*.png"))
    rain2_img_list += image_list

        
print(f"rain2 number: {len(rain2_img_list)}") 
        
for low_light_dir in  low_light_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(low_light_dir, ext))
        low_light_img_list += image_list
        
low_light_img_list = low_light_img_list * 8
        
print(f"low light number: {len(low_light_img_list)}")  
        
for haze_dir in  haze_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(haze_dir, ext))
        haze_img_list += image_list

print(f"haze number: {len(haze_img_list)}") 
        
for snow_dir in  snow_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(snow_dir, ext))
        snow_img_list += image_list

print(f"snow number: {len(snow_img_list)}") 

for blur_dir in  blur_dirs:
    for ext in exts:
        image_list = glob.glob(os.path.join(blur_dir, 'GOPR*','blur', ext))
        blur_img_list += image_list

print(f"blur number: {len(blur_img_list)}") 
        

crop_preproc = transforms.Compose([
    # transforms.CenterCrop(fix_size),
    transforms.Resize(fix_size)
    # transforms.RandomHorizontalFlip(),
])

img_preproc = transforms.Compose([
    transforms.ToTensor(),
])



print(f'The dataset length: {len(noisy_img_list)+len(jpeg_img_list)+len(rain_img_list)+len(low_light_img_list)+len(haze_img_list)+len(snow_img_list)}')

step = 0

# print("processing nosiy image")
# for lq_img_path in noisy_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("noisy", "HR")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'noisy_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'noisy_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1 
    
    
# print("processing jpg image")
# for lq_img_path in jpeg_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("jpeg", "HR")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'jpeg_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'jpeg_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1 
    
    
# print("processing rain image")   
# for lq_img_path in rain_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("rain-", "norain-")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'rain_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'rain_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1
     

# for lq_img_path in rain2_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("rainy/rain-", "gt/norain-")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'rain_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'rain_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1 
    

# print("processing low light image")
# for lq_img_path in low_light_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("low", "high")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'low_light_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'low_light_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1
     
    
# print("processing haze image")
# for lq_img_path in haze_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace(" LQ", "GT")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'haze_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'haze_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1
     
    

    
# print("processing blur image")
# for lq_img_path in blur_img_list:
#     lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

#     lq_img = Image.open(lq_img_path).convert('RGB') 
#     lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
#     gt_img_path = lq_img_path.replace("blur", "sharp")
#     gt_img = Image.open(gt_img_path).convert('RGB') 
#     gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

#     print('process {} images...'.format(step))
                
#     lq_save_path =  os.path.join(lq_path,'blur_{}.png'.format(str(step).zfill(7)))
#     gt_save_path =  os.path.join(gt_path, 'blur_{}.png'.format(str(step).zfill(7)))
     
#     lq_img.save(os.path.join(lq_save_path))
#     gt_img.save(os.path.join(gt_save_path))
#     step = step + 1 
    
step = 21083

print("processing snow image")
for lq_img_path in snow_img_list:
    lq_save_path =  os.path.join(lq_path,'{}.png'.format(str(step).zfill(7)))
    gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))

    lq_img = Image.open(lq_img_path).convert('RGB') 
    lq_img = lq_img.resize((fix_size, fix_size),Image.LANCZOS)
    
    gt_img_path = lq_img_path.replace("lq", "gt")
    gt_img = Image.open(gt_img_path).convert('RGB') 
    gt_img = gt_img.resize((fix_size, fix_size),Image.LANCZOS)

    print('process {} images...'.format(step))
                
    lq_save_path =  os.path.join(lq_path,'snow_{}.png'.format(str(step).zfill(7)))
    gt_save_path =  os.path.join(gt_path, 'snow_{}.png'.format(str(step).zfill(7)))
     
    lq_img.save(os.path.join(lq_save_path))
    gt_img.save(os.path.join(gt_save_path))
    step = step + 1


lq_img_list = glob.glob(os.path.join(lq_path, '*'))
gt_img_list = glob.glob(os.path.join(gt_path, '*'))
print(f"lq img number: {len(lq_img_list)}, gt_img_list: {len(gt_img_list)}")

    
    