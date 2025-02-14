import os
import sys
sys.path.append(os.getcwd())
import cv2

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything



from torchvision.transforms import Normalize, Compose

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian, add_gaussian_noise, add_jpg_compression, random_mixed_kernels
import random
import torch.nn.functional as F
import numpy as np

# deg_type: noisy, jpeg
# param: 50 for noise_level, 10 for jpeg compression quality
def degrade(img, deg_type, param=15):
    """
    Randomly add degradation to an image

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        degrade_types: 'noise', 'blur', 'jpeg', ...

    Returns:
        (Numpy array): Output image, shape (h, w, c), range[0, 1], float32.
    """

    if deg_type == 'noisy':
        output = add_gaussian_noise(img, sigma=param)
    elif deg_type == 'blur':
        kernel = random_mixed_kernels(['iso'], [1], kernel_size=param)
        output = cv2.filter2D(img, -1, kernel)
    elif deg_type == 'jpeg':
        output = add_jpg_compression(img, param)

    return output

    
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# deg_type: noisy, jpeg
# param: 50 for noise_level, 10 for jpeg compression quality
def generate_LQ(deg_type='jpeg', param=10):
    print(deg_type, param)
    # set data dir
    sourcedir = "/media/ps/ssd2/zyh/dataset/airnet/DIV2K/DIV2K_train_HR"
    savedir = "/media/ps/ssd2/zyh/dataset/airnet/DIV2K/DIV2K_train_jpeg"

    filepaths = []
    
    if isinstance(sourcedir, str):
        filepaths.extend(sorted([f for f in os.listdir(sourcedir) if is_image_file(f)]))
    else:
        filepaths.extend(sorted([f for f in os.listdir(sourcedir[0]) if is_image_file(f)]))
        if len(sourcedir) > 1:
            for i in range(len(sourcedir)-1):
                filepaths.extend(sorted([f for f in os.listdir(sourcedir[0]) if is_image_file(f)]))
                
    # if not os.path.isdir(sourcedir):
    #     print("Error: No source data found")
    #     exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # filepaths = [f for f in os.listdir(sourcedir) if is_image_file(f)]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename)) / 255.

        image_LQ = (degrade(image, deg_type, param) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(savedir, filename), image_LQ)
        
    print('Finished!!!')

if __name__ == "__main__":
    generate_LQ()