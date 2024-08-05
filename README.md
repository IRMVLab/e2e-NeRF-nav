**Integrating Neural Radiance Fields End-to-End for Cognitive Visuomotor Navigation**
==============================================================================================================================
This is the official implementation of paper "[**Integrating Neural Radiance Fields End-to-End for Cognitive Visuomotor Navigation**]" submitted to IEEE TPAMI. Authors: Qiming Liu, Haoran Xin, Zhe Liu and Hesheng Wang. 

This repository is still being further collated and improved, and the code and documents will be updated recently.
![](doc/network.pdf)


## Abstract & General Design

To be released.


## Prequisites
    python 3.9.8
    CUDA 12.1
    pytorch 2.2.0  
    numpy 1.26.4  
    habitat 3.0


## Usage
### Dataset
The project uses the **Habitat** simulator and **Gibson** scene dataset. Please refer to [https://github.com/facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab) for  **Habitat** installation, and the **Gibson** scene dataset can be downloaded at [https://github.com/StanfordVL/GibsonEnv#database](https://github.com/StanfordVL/GibsonEnv#database)

### Training
Train the network by running 
    
    python main.py  

Please reminder to specify the `mode`(train), `GPU`,`dataset`(path to dataset),`checkpoint_path`(path to save result) in the scripts.

The training results and saved model parameters will be saved in `checkpoint_path`.

### Testing

Please run 

    python main.py

Please reminder to specify the `mode`(test), `GPU`,`DATA_PATH`,`SCENES_DIR` and  `model_load` in the scripts.

### Results

To be released.
