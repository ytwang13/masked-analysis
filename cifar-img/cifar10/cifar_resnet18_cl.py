num_task=2
num_classes=10
# model settings
model = dict(
    type='LWFcls',
    backbone=dict(
        type='ResNet',#_CIFAR
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='incLinearClsHead',
        num_classes=int(num_classes/num_task),
        in_channels=512,
        mid_channels=None,
        # act_cfg=dict(type='BatchNorm'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_weight=3.0,
    ))


# dataset settings
dataset_type = 'CIFAR10'
data_preprocessor = dict(
    num_classes=int(num_classes/num_task),
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type='CLDataset',
        times=1,
        sub_labels=num_task,
        classes = num_classes,
        dataset=dict(
            type=dataset_type,
            data_root='data/cifar10',
            split='train',
            pipeline=train_pipeline)),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='CLDataset',
        times=1,
        sub_labels=num_task,
        classes=num_classes,
        dataset=dict(
            type=dataset_type,
            data_root='data/cifar10',
            split='test',
            pipeline=test_pipeline)),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator


optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005),
    )
optim_wrapper2 = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=2e-4),
    # paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
    )

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[60, 120, 160], gamma=0.1)
param_scheduler2 =  dict(
    type='MultiStepLR', by_epoch=True, milestones=[60, 120, 180, 220], gamma=0.1)
# dict(
    # type='MultiStepLR', by_epoch=True, milestones=[40, 50, 60], gamma=0.1)

# train, val, test setting
train_cfg = dict(type='CLEpochTrainLoop', max_epochs=200, max_task=num_task, val_interval=50) #, val_interval=1
val_cfg = dict(type='CLValLoop')
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)


# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=30, max_keep_ckpts=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=0, deterministic=False)


work_dir='/scratch/yw6594/cf/mmcl/out/cl_lwf/lr'