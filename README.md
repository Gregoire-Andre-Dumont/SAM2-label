# Sam2 label

This repository is based on [muggled sam](https://github.com/heyoeyo/muggled_sam) and let's you create accurate masks with a straightforward user-interface
 
<p align="center">
  <img src="readme_assets/create.jpg" width="500">
</p>

## Getting started

This section contains the steps that need to be taken to get started with this project and to label 
your own dataset. This project was developed on Windows 11 os on Python 3.12 and AlmaLinux 8 on Python 3.11


### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```
https://github.com/Gregoire-Andre-Dumont/SAM2-label.git
```

### 2. Install the required packages

Install the required packages (on a virtual environment is recommended) using the following command:

```shell
pip install -r requirements.txt
```

### 3. Model weights

You'll need to download the weights for SAM 2. There are four officially supported models: tiny, small, base-plus, and large. This project uses the exact same weights as the original implementations, which can be downloaded from the Model Description section of this [repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description). Note that only the checkpoint files are required. After downloading a model file, you can place it in the tm folder of this repository. Alternatively, you can keep track of the file path, as you will need to provide it when running create_references and memory_sam

