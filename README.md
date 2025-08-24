# From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data 

<!-- markdownlint-disable MD033 -->
<div align="center">

Fine-tuning stable diffusion model to generate synthetic imagery of humpback whales. 

<img src="./batchgen.png" alt="Synthetic imagery of humpback whales generated using a LoRA fine-tuned diffusion model." width="600"/>

</div>
<!-- markdownlint-enable MD033 -->

## 📑 Overview

This repository provides code and instructions for _From Noise to Nature: Diffusion Models to Advance Wildlife Detection in Remote Sensing Data_ (Sun et al., 2025) [Add link to paper when available].

The original model used to generate synthetic imagery of humpback was created by fine-tuning **Stable Diffusion 1.0 XL** using a [Hugging Face Space](https://huggingface.co/spaces/multimodalart/lora-ease) by user **multimodalart**. While the space has since been deactivated, the same fine-tuning process can be reproduced using scripts from the open-source [Diffusers](https://github.com/huggingface/diffusers) library which the HF space was running under the hood.  

## 📊 Usage

Prior to running any of these scripts, ensure your Conda environment is properly set up with all required packages installed (see *Setup* below).

Additionally, make sure you are logged into Hugging Face from your terminal.  
This is required to download and use models stored on Hugging Face.  

1. Generate a Hugging Face write token (instructions [here](https://huggingface.co/docs/hub/en/security-tokens)).  
2. Log in from the terminal using the CLI:  
   ```bash
   huggingface-cli login
3. Paste your token when prompted. 

Once logged in and paths updated, you can run the model fine-tuning or use the Jupyter notebook to generate large batches of synthetic images. Subsequently, you can run the CLIP evaluation pipeline to generate semantic similarity scores for the synthetic images.  


### 🌎 Environment Setup

To run fine-tuning with `generation/train_dreambooth_lora_sdxl.py`, we recommend using a dedicated **conda environment** with all required dependencies installed. Follow the provided documentation about how to set up a virtual environment with Diffusers [here](https://github.com/huggingface/diffusers?tab=readme-ov-file) (see *#Installation*). We recommend installing PyTorch in the same virtual environment. You can find the appropriate version of PyTorch for your machine [here](https://pytorch.org/get-started/locally/).

Activate your virtual environment, then verify installation and make sure diffusers and transformers are installed and importable: 
```bash
python -c "import diffusers, transformers; print(diffusers.__version__, transformers.__version__)"
```

After setting up your virtual environment with the necessary packages, you will also be able to run ```generation/batchgen.ipynb```.

### Fine-tuning Diffusion Model using LoRA  

- ```generation/train_dreambooth_lora_sdxl.py```  
  Python script (verbatim, from Hugging Face Diffusers library) for fine-tuning Stable Diffusion XL models. Other scripts are available for model fine-tuning in the Diffusers library as well, such as models which use different versions of Stable Diffusion (e.g SD 3.0).

- ```generation/train_dreambooth_lora_sdxl.sh```  
  Shell script that runs the train_dreambooth_lora_sdxl.py script with specified hyperparameters for fine-tuning diffusion model.   

Edit the filepaths in `generation/train_dreambooth_lora_sdxl.script.sh` to set the correct paths. Set:

   `REPOSITORY_PATH` to the path of the cloned repository, e.g., `path/to/diffusion-code`. //// maybe not needed ////  
   `CACHE_PATH` to a location where temp files can be stored during development. //// maybe not needed ////  
   `CONDA_PATH` to the path of your Conda installation. //// maybe not needed ////  
   `INSTANCE_DIR` to the location with your training images  
   `MODEL_NAME` to the location of the base diffusion model on Hugging Face  
   `OUTPUT_DIR` to the location where your fine-tuned model repository will be saved  
   `CUDA_VISIBLE_DEVICES` to 0 to use GPU; if you have a multi-GPU setup, 0 will use the GPU with index 0.  

Currently, the script is set up to run the fine-tuning the same hyperparameter values used in the (deprecated) [Hugging Face Space](https://huggingface.co/spaces/multimodalart/lora-ease) by **multimodalart**.

### Batch Synthetic Image Generation using Fine-tuned Diffusion Model  

- ```generation/batchgen.ipynb```  
  Jupyter Notebook for generating batches of synthetic images. This script uses the Diffusers library to produce large quantities of images, including those used in publication figures.  

Edit the model paths in `generation/batchgen.ipynb` to the desired model repo on Hugging Face. Adjust the number of total images and the batch size as necessary. More detailed instructions are available within the Jupyter notebook.    


### Explainability Pipeline using Contrastive Language Image Pre-Training (CLIP)

- ```evaluation/clip_evaluation.py```  

This script evaluates synthetic images against a reference set of real species images using the CLIP (Contrastive Language-Image Pre-training) model. The script computes CLIP embeddings for real species images to create a reference centroid, then scores synthetic images against this centroid using cosine similarity. This allows you to assess how well synthetic images match the visual characteristics of the target species.


#### Requirements

Install dependencies from `evaluation/requirements.txt`:
```bash
cd evaluation
pip install -r requirements.txt
```

#### Usage 
```bash
python clip_evaluation.py \
    --species-folder data/real \
    --synthetic-folder data/synthetic \
    --similarity-threshold 0.3
```

The script takes the following arguments:  
- `--species-folder`: Path to folder containing real species images (default: `data/real`)
- `--synthetic-folder`: Path to folder containing synthetic images to evaluate (default: `data/synthetic`)
- `--similarity-threshold`: Cosine similarity threshold for pass/fail (default: 0.3)

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


## 🖥️ Compute and Storage Requirements

The following compute and storage resources are recommended:

- **Compute**: We ran the scripts using a NVIDIA GeForce RTX 4060 Ti with 8GB GDDR6 GPU memory. Batch generation was also done using Google Colab's T4 GPU with 12 GB of VRAM. For the batch generation scripts, lower GPU memory can be supported by decreasing the batch size.
- **Storage**: At least ~15 GB of free disk space for the dataset, SD1.0XL model weights, and code base. Additional space may be required for model outputs and training datasets, depending if you change the patch dataset parameters and/or use your own dataset. The fine-tuned LoRA model hosted on Hugging Face occupies about 90 MB of storage. You can also change the location where model weights are stored and cached (link [here](https://huggingface.co/docs/diffusers/en/installation?install=pip) for more).

## 📂 Data Access

The LoRA fine-tuned Stable Diffusion 1.0 XL model is available for inference here: 🔗 [Drone Humpback Whale LoRA Model](https://huggingface.co/henrysun9074/drone-humpback-whale-lora-1)  

The training images and prompts are hosted on Hugging Face and can be accessed here: 🔗 [Training Dataset](doi)

All training images were collected by the **Duke Marine Robotics and Remote Sensing Lab** under NOAA permit.  

## 📝 Citations

If you use this code, the training dataset, or the model in your own research, please cite the following paper:

```bibtex
[TODO: Add bibtex citation when available]
```




