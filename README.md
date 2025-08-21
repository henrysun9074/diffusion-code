# Diffusion-Code  

Scripts and descriptions for running code used to generate images for the forthcoming publication:  
**_From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data_**  

This repository provides code and instructions to fine-tune diffusion models for generating synthetic drone imagery of humpback whales.  

---

![Batchgen](/batchgen.png)

---

## Overview  

The original model used to generate the humpback whale images above was created by fine-tuning **Stable Diffusion 1.0 XL** using a [Hugging Face Space](https://huggingface.co/spaces/multimodalart/lora-ease) by user **multimodalart**. While the space has since been deactivated, the model itself remains available for inference here:  

🔗 [Drone Humpback Whale LoRA Model](https://huggingface.co/henrysun9074/drone-humpback-whale-lora-1)  

The same fine-tuning process can also be reproduced using scripts from the open-source [Diffusers](https://github.com/huggingface/diffusers) library.  

---

## Repository Structure  

### Custom Scripts  

Located in the `custom_scripts/` directory:  

- ```train_dreambooth_lora_sdxl.py```  
  Python script (from Hugging Face Diffusers) for fine-tuning Stable Diffusion XL models.  

- ```train_dreambooth_lora_sdxl.sh```  
  Shell script that runs the Python script with the identical hyperparameters used for the original fine-tuning.  

### Training Data  

- Training images and prompts are hosted on Hugging Face:  
  🔗 [Drone Humpback Whale Training Dataset](https://huggingface.co/datasets/henrysun9074/autotrain-drone-humpback-whale-lora-1)  

All training images are property of the **Duke Marine Robots and Remote Sensing Lab**, collected under NOAA permit.  

### Image Generation  

- ```batchgen.ipynb```  
  Jupyter Notebook for generating batches of synthetic images.  
  Uses the Diffusers library to produce large quantities of images, including those used in publication figures.  

---

## Citation  

If you use this code or dataset, please cite:  
**_From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data_** (in preparation).  

---
