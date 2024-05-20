cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 ### 20min - 200 ep
# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble.py \
# --cfg-options model.head.loss_weight=0.9 model.head.loss.loss_weight=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3/.2rationew'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble.py \
# --cfg-options model.head.loss_weight=0.7 model.head.loss.loss_weight=0.3 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3/.2rationew'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble.py \
# --cfg-options model.head.loss_weight=0.5 model.head.loss.loss_weight=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3/.2rationew'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble.py \
# --cfg-options model.head.loss_weight=0.2 model.head.loss.loss_weight=0.8 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3/.2rationew'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble.py \
# --cfg-options model.head.loss_weight=0.8 model.head.loss.loss_weight=0.2  resume=True  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3/.2rationew'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble1.py \
# --cfg-options work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/all'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble1.py \
--cfg-options work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/top3'
# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_ensemble1.py \
# --cfg-options work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/all'
############# 50 EXP
# [SOFTMAX] [].SOFTMAX []SOFTMAX_KD, KD(10)🌟, 
# KD(5-wrank)  KD(10-wrank) []SOFTMAX(10-wrank) ,[]SOFTMAX(5+ -wrank) 
# []SOFTMAX_KD(10-wrank) [SOFTMAX](10-wrank) kd(10-wrank) []SOFTMAX_KD(10-wrank)
# []KD(10-wrank)sche 0.2  []KD(5+-wrank)sche 0.2  []KD(+5-wrank)sche 0.2  
# []KD(+5-wrank)sche 0.2 + ce []KD(5+-wrank)sche 0.2 + ce  []KD(5+-wrank) + ce 
# []KD(5+-wrank) + ce 200ep []KD(top3-wrank) + ce 200ep []KD(top3-wrank) sche 0.2+ ce 200ep
# []KD(top3-wrank) sche 0.2+ ce 200ep ratio?


# ?base-c40 [50, 70, 200] wrank)  base-c36 200 wrank)
##### ratio log
# ｜ kd  ｜ ce  | acc  | rank |
# ｜ 0.9 ｜ 0.1 | acc  | rank |
# ｜ 0.8 ｜ 0.2 | acc  | rank |
