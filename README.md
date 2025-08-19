# diffusion-code
Scripts and descriptions for running code used to generate images for forthcoming publication: "From Noise to Nature: Diffusion Models for Wildlife Detection in Remote Sensing Data".  

This repository houses code and instructions to fine-tune diffusion models to generate synthetic drone imagery of humpback whales.

![Batchgen](/batchgen.png)

The original model used to generate the above humpback whale images produced by fine-tuning Stable Diffusion 1.0 XL using a [space](https://huggingface.co/spaces/multimodalart/lora-ease) by user 'multimodalart' on Hugging Face that has now been deactivated. The model is available on Hugging Face for inference [here](https://huggingface.co/henrysun9074/drone-humpback-whale-lora-1). While the original space is no longer active, the same fine-tuning process can be done with scripts from the open-source [Diffusers](https://github.com/huggingface/diffusers) library.  

Within the */custom_scripts/* directory in this repo, we include the Python script provided by Diffusers for fine-tuning Stable Diffusion XL models (**train_dreambooth_lora_sdxl.py**). The shell script, **train_dreambooth_lora_sdxl.sh**, runs the Python script with identical hyperparameter values to those used for model fine-tuning. 

The training images and prompts used to fine-tune the humpback whale LoRA model are all available on Hugging Face [here](https://huggingface.co/datasets/henrysun9074/autotrain-drone-humpback-whale-lora-1). Training images are property of the Duke Marine Robots and Remote Sensing Lab under permit by NOAA.

Following model fine-tuning, or to work with an existing fine-tuned LoRA model, we used **batchgen.ipynb** to generate batches of synthetic images. This Jupyter Notebook uses the Diffusers library to produce large quantities of synthetic images at a time, with all whale images in publication figures sourced from batch-generated outputs with the aforementioned humpback whale LoRA.