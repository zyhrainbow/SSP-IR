# export HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT=https://hf-mirror.com


CUDA_VISIBLE_DEVICES="0" accelerate  launch  --main_process_port 29497  train.py \
--pretrained_model_name_or_path="/media/ps/ssd2/zyh/video/checkpoints/stable-diffusion-v1-5" \
--output_dir="./experiment/20250214" \
--root_folders '/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_lsdir_20k,/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_for_seesr,/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_ffhq,/media/ps/ssd2/zyh/dataset/sr_realesrgan/training_ost' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=16 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=100 \
--max_train_steps=100000 \