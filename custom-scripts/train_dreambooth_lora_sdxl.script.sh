#!/bin/bash

. /c/Users/hs325/AppData/Local/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

'''
Change the first three variables below to match the folders for the files you are using
MODEL_NAME should be the repo with the base diffusion model on Hugging Face
INSTANCE_DIR should correspond to the folder with input images for fine-tuning
OUTPUT_DIR is where the resulting fine-tuned model repo will be placed
'''
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="D:\Projects\HappyHumpbacks\TrainDreamboothHWIMGs"
export OUTPUT_DIR="D:\Projects\HappyHumpbacks\HHlora-out"
export CUDA_VISIBLE_DEVICES=0

'''
The following code runs the script. Adjust existing hyperparameters and include more optional ones as necessary
'''

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="Drone image of a <s0><s1> in the ocean, clear water, visible pectoral fins, ultra realistic" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-08 \
  --adam_weight_decay=0.0001 \
  --optimizer="prodigy" \
  --prodigy_decouple \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --max_grad_norm=1.0 \
  --max_train_steps=1350 \
  --cache_latents \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac=0.5 \
  --rank=16 \
  --sample_batch_size=4 \
  --num_validation_images=4 \
  --validation_epochs=50 \
  --seed=42 \
  --logging_dir="logs" \
  --report_to="tensorboard"
