
# --process_size 512 \
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" python test.py \
--pretrained_model_path "/media/ps/ssd2/zyh/video/checkpoints/stable-diffusion-v1-5" \
--prompt None \
--seesr_model_path /media/ps/ssd2/zyh/lv/emotion/SeeSR/experiment/Ours/checkpoint-100000 \
--image_path  ./data \
--output_dir results/ours1 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--process_size 512 \
#