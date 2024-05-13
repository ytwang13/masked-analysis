## Introduction
in this repo, we provide code for project " Investigating ResNet performance under parameter constraint situation". Here ew will provide guidance for installation, peforming experiments and some results.

## Install 

Please refer to MMpretrain for more details, below are the copy of theirs:

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

For multi-modality models support, please install the extra dependencies by:

```shell
mim install -e ".[multimodal]"
```
---

## Experiments
All experiments can be seen in tools/res_trails directory
### Baseline- architecture modification
Follow the scripts in run_baseline.sh

```shell
 sh tools/res_trails/run_baseline.sh
```
### Ensemble, etc.
Follow the scripts in run_ensemble.sh

```shell
 sh tools/res_trails/run_ensemble.sh
```
### Normalization, Activation, Dropout, Masking
Follow the scripts in run_cfgs.sh

```shell
 sh tools/res_trails/run_cfgs.sh
```

#### Results, and evaluation.

```shell
 sh tools/res_trails/run_evaluation.sh
```