import random
import shutil
import os
import glob

# lr_paths = "/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir/lr/"
gt_paths = "/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir/gt/"
llm_caption_paths = "/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir/llm_caption"
sr_paths = "/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir/sr_bicubic/"

# lr_list = glob.glob(os.path.join(lr_paths, '*.png'))
# sample_size = 20000
# random_sample = random.sample(lr_list, sample_size)

# # 创建新的文件夹用于保存抽样图像
# lr_output_folder = '/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k/lr/'
# os.makedirs(lr_output_folder, exist_ok=True)
# gt_output_folder = '/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k/gt/'
# os.makedirs(gt_output_folder, exist_ok=True)
# llm_caption_output_folder = '/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k/llm_caption/'
# os.makedirs(llm_caption_output_folder, exist_ok=True)

# # 将抽样图像复制到新的文件夹中
# for lr_path in random_sample:
#     basename = os.path.basename(lr_path)
#     output_path = os.path.join(lr_output_folder, basename)
#     shutil.copyfile(lr_path, output_path)
#     gt_path = os.path.join(gt_paths, basename)
#     output_path = os.path.join(gt_output_folder, basename)
#     shutil.copyfile(lr_path, output_path)
#     llm_caption_path = os.path.join(llm_caption_paths, basename.replace('.png', '.txt'))
#     output_path = os.path.join(llm_caption_output_folder, basename.replace('.png', '.txt'))
#     shutil.copyfile(lr_path, output_path)

lr_list = glob.glob(os.path.join("/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k/lr", '*.png'))
output_folder = '/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k/gt/'
os.makedirs(output_folder, exist_ok=True) 
for lr_path in lr_list:
    basename = os.path.basename(lr_path)
    sr_path = os.path.join(gt_paths, basename)
    output_path = os.path.join(output_folder, basename)
    shutil.copyfile(sr_path, output_path)  
    # file = open(sr_path, 'r')
    # validation_prompt = file.read()
    # file.close()
    # file = open(output_path, "w")
    # file.write(validation_prompt)
    # file.close()

