# Pytorch Code for "MAP-based Problem-Agnostic Diffusion Model for Inverse Problems"

[MAP-based Problem-Agnostic Diffusion Model for Inverse Problems](https://arxiv.org/abs/2501.15128)


## Abstract

Diffusion models have indeed shown great promise in solving inverse problems in image processing. In this paper, we propose a novel, problem-agnostic diffusion model called the maximum a posteriori (MAP)-based guided term estimation method for inverse problems. To leverage unconditionally pretrained diffusion models to address conditional generation tasks, we divide the conditional score function into two terms according to Bayes' rule: an unconditional score function (approximated by a pretrained score network) and a guided term, which is estimated using a novel MAP-based method that incorporates a Gaussian-type prior of natural images. This innovation allows us to better capture the intrinsic properties of the data, leading to improved performance. Numerical results demonstrate that our method preserves contents more effectively compared to state-of-the-art methods-for example, maintaining the structure of glasses in super-resolution tasks and producing more coherent results in the neighborhood of masked regions during inpainting. Our numerical implementation is available at https://github.com/liuhaixias1/MAP-DIFFUSION-IP.
-----------------------------------------------------------------------------------------

## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1 (other version is also fine)


## Getting started 



### Step 1: Set environment

Create a new environment and install dependencies

```
conda create -n MAP python=3.8

conda activate MAP

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

If you fail to install mpi4py using the pip install, you can try conda as follows
```
conda install mpi4py
```

In addition, you might need 

```
pip install scikit-image
pip install blobfile
```

Finally, make sure the code is run on GPU, though it can run on cpu as well.  


### Step 2:  Download pretrained checkpoint
For FFHQ, download the pretrained checkpoint "ffhq_10m.pt"  from  [link_ffhq_checkpoint](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), and paste it to ./models/


### Step 3:  Prepare the dataset
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set. We also provide other demo data samples in ./data/ used in our paper.

### Step 4: Perform Posterior Sampling for different tasks 

```
python3 main.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
--save_dir './saved_results'
```
## You may need to change parameters for coffa, coffb and coffc, where coffa and coffb are set in the file diffusion_config.yaml, coffc is in guided_diffusion/condition_methods.py


## Possible model configurations

```
- configs/model_config.yaml 

```


## Possible task configurations
```
# Various linear inverse problems
- configs/sr4_config.yaml
- configs/denoise_config.yaml
- configs/inpainting_config.yaml

```


## Citation 
If you find the code useful for your research, please consider citing as 

```
@article{tao2025map,
  title={MAP-based Problem-Agnostic Diffusion Model for Inverse Problems},
  author={Pingping Tao and Haixia Liu and Jing Su and Xiaochen Yang and Hongchen Tan},
  journal={arXiv preprint arXiv:/2501.15128},
  year={2025}
}
```


## References

This repo is developed based on [DMPS code](https://github.com/mengxiangming/dmps). Please also consider citing it if you use this repo. 
```
@article{meng2022diffusion,
  title={Diffusion Model Based Posterior Samplng for Noisy Linear Inverse Problems},
  author={Meng, Xiangming and Kabashima, Yoshiyuki},
  journal={arXiv preprint arXiv:2211.12343},
  year={2022}
}

```
