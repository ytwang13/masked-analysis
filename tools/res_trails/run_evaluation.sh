##########################################################################################
#################################### MODEL EVAL EXP -get parameters, submit kaggle results
##########################################################################################


################################# get parameters, replace -your-config-path- to your config path
python tools/analysis_tools/get_flops.py -your-config-path- --shape 32

################################ submit kaggle results
#### use the corresponding config for submission, 
#### just remember to change dataset to CIFARTEST, 
#### and put the downloaded data file in the same data/cifar10/cifar-10-batches-py directory.
# also use the corresponding checkpoint path
# --out defines the output path
python tools/test.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c32_kaggle.py \
/scratch/yw6594/cf/mmcl/out/dl/res18-c10/cifarbase/c32/epoch_200.pth \
--out-item 'pred' --out /scratch/yw6594/cf/mmcl/out/dl/res18-c10/kaggle.pkl --work-dir '/scratch/yw6594/cf/mmcl/out/dl/res18-c10'