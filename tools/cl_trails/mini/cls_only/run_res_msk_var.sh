cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
mask_ratio=${1:-0.13}
inv_mode=${2:-'v2'}
mask_mode=${3:-'sgf'}

# model.head.mask_inv=True model.head.mask_ratio=$mask_ratio
# model.head.mask_mode=$mask_mode model.head.mask_mode=$mask_mode agf sgf
# model.head.mask_inv=True
python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=7.0 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/7.0'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=5.0 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/5.0'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=1.0 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/1.0'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.5 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.5'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.3 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.3'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.1 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.1'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.090 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.090'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.050 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.050'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.030 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.030'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskclso.py \
--cfg-options model.head.mask_mode=$mask_mode model.head.mask_inv=True model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.010 model.head.inv_mode=$inv_mode work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/clsomaskinvold'${inv_mode}'/KDmaskweight'${mask_mode}'/'${mask_ratio}'/0.010'



#woinv maskratio 0.85 0.20 0.60 0.50 0.35 0.15 0.20 ❓here have a issue without inv all use kd loss！！！！

#invold maskratio 0.13 0.10 0.20 
###### now use array to expand over mask_ratio
#### v1 +results 🛫
# involdarray others default| 0.95 0.85 0.75 0.50 0.35 0.25 0.15 [0.35 .25 .15 .85 |5.0 1.0]
#### v2 results 🛫
# involdarray others default| 0.95 0.85 0.75 0.50 0.35 0.25 0.15 [0.15 | 5.0 ]

###### mask_loss?
# l2[not so working rankme collapse] now try much much smaller loss_weight <0.1
# .5 .3 .1 .09 .07 .05 .03 .01 lossweight
#### v1 +results  🛫
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 （70rankme but slightly lower acc）
#### v2 +results  🛫
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 （70rankme but slightly lower acc）

# l2[try after softmax?] [softmax]'sgf' mode 🛫 ###### mask_mode? ###### 
#### v1 +results
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 
#### v2 +results🛫
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 
# [softmax]'sgf' mode 
#### #### #### ####  v1 +results
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 🛫
#### v2 +results
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 🛫


# 'sgf'-previous  # 'sgf'mode here l2 use small l2 ###### mask_mode? ###### 
#### v1 +results -l2 🛫
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 
#### v2 +results 🛫
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 
# [softmax]'sgf' mode 
################### the softmax do not have some significant improvement... #####🛫
#### #### #### ####  v1 +results
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 🛫
#### v2 +results
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8 🛫

######  schedular?
# test method start base=1 (here test loss and mode?) 6exps?
# test method start base=25 (here test loss and mode?) 6exps?
## l2 - a/s v1
## cs - a/s
## kd - a/s

# move to ema then?

