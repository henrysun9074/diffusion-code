# From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data 

<!-- markdownlint-disable MD033 -->
<div align="center">

Fine-tuning stable diffusion model to generate synthetic imagery of humpback whales. 

<img src="./batchgen.png" alt="Synthetic imagery of humpback whales generated using a LoRA fine-tuned diffusion model." width="600"/>

</div>
<!-- markdownlint-enable MD033 -->

## 📑 Overview

This repository provides code and instructions for _From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data_ (Sun et al., 2025) [Add link to paper when available].

The original model used to generate synthetic imagery of humpback was created by fine-tuning **Stable Diffusion 1.0 XL** using a [Hugging Face Space](https://huggingface.co/spaces/multimodalart/lora-ease) by user **multimodalart**. While the space has since been deactivated, the same fine-tuning process can also be reproduced using scripts from the open-source [Diffusers](https://github.com/huggingface/diffusers) library.  

## 📊 Usage

Prior to running any of these scripts, ensure your Conda environment is properly set up with all required packages installed (see *Setup* below).

Additionally, make sure you are logged into Hugging Face from your terminal.  
This is required to download and use models stored on Hugging Face.  

1. Generate a Hugging Face write token (instructions [here](https://huggingface.co/docs/hub/en/security-tokens)).  
2. Log in from the terminal using the CLI:  
   ```bash
   huggingface-cli login
3. Paste your token when prompted. 

Once logged in and paths updated, you can run the model fine-tuning or use the Jupyter notebook to generate large batches of synthetic images. These scripts are all located in the `custom_scripts/` directory:  

### Fine-tuning Diffusion Model using LoRA  

- ```train_dreambooth_lora_sdxl.py```  
  Python script (verbatim, from Hugging Face Diffusers library) for fine-tuning Stable Diffusion XL models. Other scripts are available for model fine-tuning in the Diffusers library as well, such as models which use different versions of Stable Diffusion (e.g SD 3.0).

- ```train_dreambooth_lora_sdxl.sh```  
  Shell script that runs the train_dreambooth_lora_sdxl.py script with specified hyperparameters for fine-tuning diffusion model.   

Edit the filepaths in `custom_scripts/train_dreambooth_lora_sdxl.script.sh` to set the correct paths. Set:

   `REPOSITORY_PATH` to the path of the cloned repository, e.g., `path/to/diffusion-code`. //// maybe not needed ////  
   `CACHE_PATH` to a location where temp files can be stored during development. //// maybe not needed ////  
   `CONDA_PATH` to the path of your Conda installation. //// maybe not needed ////  
   `INSTANCE_DIR` to the location with your training images  
   `MODEL_NAME` to the location of the base diffusion model on Hugging Face  
   `OUTPUT_DIR` to the location where your fine-tuned model repository will be saved  
   `CUDA_VISIBLE_DEVICES` to 0 to use GPU; if you have a multi-GPU setup, 0 will use the GPU with index 0.  

Currently, the script is set up to run the fine-tuning the same hyperparameter values used in the (deprecated) [Hugging Face Space](https://huggingface.co/spaces/multimodalart/lora-ease) by **multimodalart**.

### Batch Synthetic Image Generation using Fine-tuned Diffusion Model  

- ```batchgen.ipynb```  
  Jupyter Notebook for generating batches of synthetic images. This script uses the Diffusers library to produce large quantities of images, including those used in publication figures.  

Edit the model paths in `custom_scripts/batchgen.ipynb` to the desired model repo on Hugging Face. Adjust the number of total images and the batch size as necessary. More detailed instructions are available within the Jupyter notebook.  

## 📦 Installation

### Option 1: Using Git (Recommended)

1. Clone the repository:

   ```bash
   git clone https://github.com/henrysun9074/diffusion-code.git
   cd diffusion-code
   ```

### Option 2: Downloading as a ZIP file

1. Download the repository as a ZIP file from this repo's GitHub page using the green "Code" button  at the top of the webpage and selecting "Download ZIP".

   Alternatively, you can download it directly using the command line:

   ```bash
   wget https://github.com/henrysun9074/diffusion-code.git
   ```

2. Extract the ZIP file:

   ```bash
   unzip diffusion-code.zip
   cd diffusion-code
   ```

### 🌎 Environment Setup

Create a virtual environment using environment.yml file and install all required packages
- need to add yaml file with dependencies 
- add instructions on how to build environment using conda
- e.g. conda create....using yml


## 🖥️ Compute and Storage Requirements

The following compute and storage resources are recommended:

- **Compute**: X x NVIDIA AX GPU or better with at least XGB of free memory. The training and inference scripts are designed to run on .... , but can be adapted for local use?
- **Storage**: At least X GB of free disk space for the dataset, model weights, and code base. Additional space may be required for model outputs and training datasets, depending if you change the patch dataset parameters and/or use your own dataset.

## 📂 Data Access

The LoRA fine-tuned Stable Diffusion 1.0 XL model is available for inference here: 🔗 [Drone Humpback Whale LoRA Model](https://huggingface.co/henrysun9074/drone-humpback-whale-lora-1)  

The training images and prompts are hosted on Hugging Face and can be accessed here: 🔗 [Training Dataset](doi)

All training images were collected by the **Duke Marine Robotics and Remote Sensing Lab** under NOAA permit.  

## 📝 Citations

If you use this code, the training dataset, or the model in your own research, please cite the following paper:

```bibtex
[TODO: Add bibtex citation when available]
```




