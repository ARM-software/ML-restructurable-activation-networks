# SPDX-License-Identifier: BSD-3-Clause

conda create -n RANi_env -y
source activate RANi_env

conda install -c conda-forge pycocotools -y
conda install -c conda-forge timm -y #installs torch and torchvision automatically
conda install -c conda-forge tqdm -y
pip install ptflops tensorboardX six
