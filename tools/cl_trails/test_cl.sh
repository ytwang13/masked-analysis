cd  /scratch/yw6594/cf/mmcl
# export CUDA_VISIBLE_DEVICES=2

######### test CLdataset
# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py
# for i in {1..2}
# do
# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=0. work_dir='/scratch/yw6594/out/cl_test/test_multi/single'

#     python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msk.py \
#     --cfg-options model.head.loss_weight=0. work_dir='/scratch/yw6594/out/cl_test/test_setting/single'
# done
#######################
#### single time ~0.015
#### parallel time ~0.02
######################################################## TODO
####### 👄write imageclassifier to do distill and mask
###### 👄write loop to help change or do a hook? x
###### 🦵mask loss with a complement mask forward?
######  👄 write rankme as a metric to present
######   👄 write ensemble code
# model.head.mid_channels=[128]
# model.head.loss_weight=1.0
# model.neck.p=3
# model.head.dropout=0.1
# model.head.act_cfg.negative_slope=0.1

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_kd.py \
# --cfg-options model.head.loss_weight=0.05 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd5_start25'  #conv5x5  bn256


# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.act_cfg.negative_slope=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/alllrelu'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.act_cfg.negative_slope=0.3 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/alllrelu'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.act_cfg.negative_slope=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/alllrelu'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.act_cfg.negative_slope=0.6 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/alllrelu'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.act_cfg.negative_slope=0.8 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/alllrelu'


# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.3 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.7 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
# --cfg-options model.head.dropout=0.9 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/allrelu-dropout/lratio0.005'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py \
--cfg-options work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/base/rank32'

python tools/analysis_tools/get_flops.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32.py --shape 32
############ MODEL SIZE LOG
##### base -c40 4.384M

#################################################################################### kaggle test code 🛫🛫🛫🛫🛫🛫🛫
# python tools/analysis_tools/get_flops.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32_kaggle.py --shape 32

# python tools/test.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c32_kaggle.py \
# /--your-own-dir/out/dl/res18-c10/cifarbase/c32/epoch_200.pth \
# --out-item 'pred' --out /--your-own-dir/out/dl/res18-c10/kaggle.pkl --work-dir '/--your-own-dir/out/dl/res18-c10'

# python tools/train.py /--your-own-dir/cifar-img/draft/cifar10_resnet18.py \
# --cfg-options work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c16/lr_mult/b0.99'
# python tools/analysis_tools/get_flops.py /--your-own-dir/cifar-img/draft/cifar10_resnet18.py --shape 32

########## we test baseline resnet18 (7x7 conv1 11.182M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 85.6 | 99.3 |
# |70   | 86.4 | 99.3 |
# |200  | 87.5 | 99.3 |


##########  we test baseline resnet18-cifar (3x3 conv1 11.174M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.8 | 99.8 |
# |70   | 93.5 | 99.8 |
# |200  | 94.4 | 99.8 |

        # 50 &  40  & 4.384  & 92.29 & 50 & 40 & 4.393    & 91.97 \\
        # 70 &  40  & 4.384  & 92.90 & 70 & 40 & 4.393    & 92.69 \\
        # 200 & 40 & 4.384   & 94.14  & 200 & 40 & 4.393  & 94.16 \\

########## we test baseline resnet18-cifar (3x3 base_c=40 😊4.384M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.29| 99.79|
# |70   | 92.90| 99.83|
# |200  | 94.14| 99.78| ### so far sota? 🛫🛫🛫🛫🛫🛫🛫🛫

########## we test baseline resnet18-cifar (3x3 base_c=40 w2x2 gap😊4.39MM) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.46| 99.76|
# |70   | 92.92| 99.77|
# |200  | 94.06| 99.68| 

#### dropout conv3x3                                                                      
# |50   | 92.42| 99.74| 0.1
# |50   | 92.07| 99.78| 0.3
# |50   | 92.42| 99.83| 0.5
# |50   | 92.10| 99.76| 0.6
# |50   | 91.95| 99.77| 0.8
#### dropout conv3x3  + dense -256 4.708
# |50   | 92.25| 99.74| 0.1
# |50   | 92.37| 99.78| 0.3
# |50   | 92.35| 99.83| 0.5
# |50   | 91.59| 99.76| 0.6
# |50   | 91.67| 99.77| 0.8



### conv5x5 4.393M maybe 5x5 is sufficient for 32x32?
# |epoch| acc-1| acc-5|
# |50   | 84.78| 99.28| conv7x7?
# |50   | 91.97| 99.78|
# |70   | 92.69| 99.76|
# |200  | 94.16| 99.68|

########## we test baseline resnet18-cifar (3x3 base_c=40 gem😊 4.381M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.30| 99.82| p=3
# |50   | 92.22| 99.76| p=2
# |50   | 92.49| 99.73| p=1
# |50   | 92.28| 99.73| p=5
# |50   | 91.95| 99.80| p=6
# |50   | 91.84| 99.80| p=7

# |70   | 92.92| 99.77|x
# |200  | 94.06| 99.68| 



# kd
##### methodep=10 not so well, around89 90
# |50   | 92.29| 99.79| weight 1.0
# |50   | 92.29| 99.79| weight 1.0
# |50   | 92.29| 99.79| weight 1.0
# |50   | 92.29| 99.79| weight 1.0

####densehead
# |50   | 92.44| 99.83| wdensehead-1024 wrelu 4.716M
# |50   | 92.47| 99.75| wdensehead-512 wrelu  
# |50   | 92.61| 99.80| wdensehead-256 wrelu   ### so far sota? 🛫
# |50   | 92.50| 99.79| wdensehead-128 wrelu  

# |50   | 91.66| 99.73| wdensehead-1024 wrelu 4.716M wBN
# |50   | 92.16| 99.73| wdensehead-512 wrelu  
# |50   | 92.41| 99.80| wdensehead-256 wrelu  
# |50   | 92.57| 99.81| wdensehead-128 wrelu  

# |50   | 92.11| 99.80| wdensehead-1024 wrelu 4.716Mdropout-0.6
# |50   | 92.45| 99.87| wdensehead-512 wrelu  dropout-0.6
# |50   | 92.29| 99.80| wdensehead-256 wrelu  dropout-0.6
# |50   | 91.75| 99.77| wdensehead-128 wrelu  dropout-0.6

################ 5x5
# |50   | 92.26| 99.71|
# |70   | 92.72| 99.79|
# |200  | 94.17| 99.82| 4.384M


########## we test baseline resnet18-cifar (base_c=32 😊2.81M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 91.96| 99.78|
# |70   | 92.83| 99.8 |
# |200  | 93.76 | 99.8 | /--your-own-dir/out/dl/res18-c10/cifarbase/c32/epoch_200.pth ### so far sota? 🛫🛫🛫🛫🛫🛫🛫🛫

########## we test baseline resnet18-cifar (base_c=16 😊0.71M) for 50 70 200 overall epochs
# |50   | 90.33| 99.73| c16 0.711M [25, 40]
# |50   | 90.42| 99.75| c16 0.711M [25, 35]
# |50   | 90.0600| 99.6900| c16 0.711M [25, 35, 45]
# |50   | 89.7100| 99.7000| c16 0.745M +mid 128-256
# |50   | 90.1800| 99.7300| c16 0.746M +mid 128-256 +BN
# |50   | 90.1400| 99.7300| c16 0.746M +mid 128-256 +BN [25, 35]
# |50   | 90.39  | 99.7200| c16 0.718M +mid 128-64
# |50   | 90.03  | 99.6900| c16 0.718M +mid 128-64 +BN
# |50   | 88.22  | 98.7500| c16 0.71M +mid 128-4
# |50   | 89.19  | 99.3500| c16 0.71M +mid 128-4 + BN
        # 50 & 40 & 4.746  & 89.71 & mid 256 & 40 & 4.746   & 90.03 \\
        # 50 & 40 & 4.718  & 90.18 & mid 64 & 40 &  4.718    & 88.22 \\
        # 50 &40 &  4.710  & 90.14  & mid  4 & 40 &  4.710   &  89.19 \\
# |70   | 90.53  | 99.6700| c16 0.711M [35, 50]
# |200  | 91.98  | 99.7300| c16 0.711M [100, 150] 


##### can we surpass this upperbound?
############### lr not good
# |50   |  89.95 | 99.64 | c16 0.711M [25, 35] head0.1_lr 5min
# |50   |  90.05 | 99.72 | c16 0.711M [25, 35] head0.5_lr 5min
# |50   |  90.54 | 99.72 | c16 0.711M [25, 35] head0.9_lr 5min
# |50   |  90.07 | 99.66 | c16 0.711M [25, 35] head0.99_lr 5min
# |50   |  85.00 | 99.41 | c16 0.711M [25, 35] back0.1_lr 5min
# |50   |  90.21 | 99.69 | c16 0.711M [25, 35] back0.9_lr 5min
# |50   |  89.98 | 99.61 | c16 0.711M [25, 35] back0.99_lr 5min

############## BN for c32
# |50   |  92    | 99.79 | c32 2.813M conv5x5 bn128  92.02 b.99
# |70   |  92    | 99.79 | c32 2.813M conv5x5 bn128  92.32 b.99
# |200  |  93.55 | 99.80 | c32 2.813M conv5x5 bn128  93.81 b.99

############## densenet C32
# |50   |  91.66 | 99.74 | c32 3.086M conv5x5 bn128 -1024
# |50   |  90.89 | 99.71 | c32 3.086M conv5x5 bn128 -1024 wd2e-5
# |200  |  93.76 | 99.83 | c32 3.086M conv5x5 bn128 -1024
# |200  |  93.75 | 99.81 | c32 3.086M conv5x5 -1024 wrelu
# |200  |  93.36 | 99.81 | c32 3.086M conv5x5 bn128 -1024 wrelu
# |50   |  91.7  | 99.74 | c32 >2.813M conv5x5 bn128 -512
# |200  |  93.4  | 99.71 | c32 3.086M conv5x5 bn128 -512
# |200  |  93.73 | 99.81 | c32 3.086M conv5x5 -512 wrelu 🌟🌟
# |200  |  93.67 | 99.75 | c32 3.086M conv5x5 bn128 -512 wrelu 🌟🌟
# |50   |  91.95 | 99.76 | c32 >2.813M conv5x5 bn128 -256 🌟
# |200  |  93.64 | 99.73 | c32 3.086M conv5x5 bn128 -256
# |200  |  93.84 | 99.84 | c32 3.086M conv5x5 -256 wrelu
# |200  |  93.43 | 99.72 | c32 3.086M conv5x5 bn128 -256 wrelu

# |50   |  91.41 | 99.76 | c32 >2.813M conv5x5 bn128 -256 h.5
# |50   |  91.01 | 99.76 | c32 >2.813M conv5x5 bn128 -256 b.5
# |50   |  91.6  | 99.76 | c32 >2.813M conv5x5 bn128 -256 b.9
# |50   |  91.37  | 99.76 | c32 >2.813M conv5x5 bn128 -256 [25, 35, 40]
# |50   |  91.49  | 99.76 | c32 >2.813M conv5x5 nobn128 -256 [25, 35]



# |50   |  91.62 | 99.70 | c32 2.813M conv5x5 bn128 -128
# |200  |  93.70 | 99.78 | c32 3.086M conv5x5 bn128 -128
# |200  |  93.89 | 99.81 | c32 3.086M conv5x5 bn128 -128

# |50   |  91.83 | 99.77 | c32 2.813M conv5x5 bn128 -64
# |50   |  91.68 | 99.76 | c32 2.813M conv5x5 bn128 -32
# |50   |  91.5  | 99.56 | c32 2.813M conv5x5 bn128 -4



