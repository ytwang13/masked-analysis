import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.dataset.base_dataset import BaseDataset

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner.loops import *
from mmengine.runner.runner import *
from mmengine.registry import LOOPS, DATA_SAMPLERS, DATASETS, HOOKS

from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)


## CLDATASET
import copy

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init


@DATASETS.register_module()
class CLDataset:
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 sub_labels: list,
                 times: int,
                 classes: int,
                 task_mode: str ='cl',
                 seed: int=0,
                 lazy_init: bool = False):
        super().__init__()
        self.seed = seed
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self.times = times
        self._metainfo = self.dataset.metainfo
        ### Handle sub_label here
        self.sub_labels = sub_labels
        self.sub_indices = []
        self.CLASSES = [self.dataset.CLASSES[sub_label] for sub_label in self.sub_labels]
        for index in range(len(self.dataset)):
            label = self.dataset.get_data_info(index)['gt_label']
            if label in sub_labels:
                self.sub_indices.append(index)
        self._ori_len = len(self.sub_indices)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()


    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        
        self.dataset.full_init()
        np.random.seed(self.seed)
        new_order = np.random.permutation(len(self.dataset.CLASSES))
        data_list = self.dataset.data_list
        for data in data_list:
            data['gt_label'] = int(new_order[data['gt_label']])
        self._ori_len = len(self.sub_indices)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx: Global index of ``RepeatDataset``.

        Returns:
            idx (int): Local index of data.
        """
        return self.sub_indices[idx % self._ori_len]

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate the '
                'speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset[sample_idx]

    @force_full_init
    def __len__(self):
        return self.times * self._ori_len

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`RepeatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `RepeatDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`RepeatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `RepeatDataset`.')
    
    # @force_full_init
    # def get_gt_labels(self):
    #     """Get all ground-truth labels (categories).

    #     Returns:
    #         np.ndarray: categories for all images in the task.
    #     """
    #     # gt_labels = np.array([data['gt_label'] for data in self.data_infos])
    #     if self.task_mode == 'cl' or 'cl' in self.task_mode:
    #         gt_labels = np.array([data['gt_label'] for data in self.dataset.data_infos])
    #     else:
    #         gt_labels = np.array([data['gt_label']-min(self.sub_labels) for data in self.dataset.data_infos])
    #     return gt_labels



@LOOPS.register_module()
class CLEpochTrainLoop(EpochBasedTrainLoop):
    """Loop for Continual learning
    All args same as EpochBasedRunner
    - change

    """
    def __init__(
            self,
            runner: Runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            max_task: int = 2,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        self._runner = runner
        # self.runner = dataloader
        self.dataloaders = self.build_cldataloader(dataloader)
        self.task_id = 1
        self.max_task = max_task
        dataloader = self.dataloaders[0]
        super().__init__(runner, dataloader,  max_epochs, val_begin, val_interval, dynamic_intervals)
        
    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        # for 
        while self.task_id < self.max_task + 1:
            while self._epoch < self._max_epochs and not self.stop_training:
                self.run_epoch()

                self._decide_current_val_interval()
                if (self.runner.val_loop is not None
                        and self._epoch >= self.val_begin
                        and self._epoch % self.val_interval == 0):
                    self.runner.val_loop.run()## here pass self._Epoch
            self.runner.logger.info(f'🛫🛫🛫🛫🛫🛫🛫🛫 Now finished task_{self.task_id} 😊😊😊😊😊😊😊😊')
            if self.task_id < self.max_task: 
                self.cl_configure()
            self.task_id +=1
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper, mode='lwf')

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def run_epoch(self):
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            datas = data_batch['data_samples']
            for data in datas:
                data.gt_label = data.gt_label-int(min(self.dataloader.dataset.sub_labels))
            data_batch['data_samples'] = datas
            self.run_iter(idx, data_batch)
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
    
    def cl_configure(self):
        ######################## prepare for the next task
        self.dataloader = self.dataloaders[self.task_id]
        if hasattr(self.runner.model, 'module'):
            self.runner.model.module.cl_init()
            self.runner.model.module.head.add_head(self.num_class)
        else:
            self.runner.model.cl_init()
            self.runner.model.head.add_head(self.num_class)
        optim_cfg = self.runner.cfg.get('optim_wrapper2')
        self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)
        self.runner.param_schedulers = self.runner.build_param_scheduler(self.runner.cfg.get('param_scheduler2'))
        self._epoch = 0
        self._iter = 0
        if self.task_id == 1:
            self._max_epochs = int(self._max_epochs * 1.25)
            self._max_iters = self._max_epochs * len(self.dataloader)


    def build_cldataloader(self, dataloader):
        diff_rank_seed = self.runner._randomness_cfg.get(
            'diff_rank_seed', False)
        seed = self.runner.seed
        if isinstance(dataloader, DataLoader):
                return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        datasets = []
        if isinstance(dataset_cfg, dict):
            sub_labels = []
            task_num = dataset_cfg['sub_labels']
            class_per_task = int(dataset_cfg['classes']/ task_num)
            self.num_class = class_per_task
            # np.random.seed(seed)
            # new_order = np.random.permutation(dataset_cfg['classes'])
            for task_id in range(task_num):
                org_labels = list(np.array(range(class_per_task))+class_per_task*task_id)
                sub_labels.append(org_labels)
                # sub_labels.append(list(map(lambda x:new_order[x],org_labels)))
            for sub_label in sub_labels:
                dataset_cfg.sub_labels = sub_label
                dataset_cfg.seed = seed
                dataset = DATASETS.build(dataset_cfg)
                if hasattr(dataset, 'full_init'):
                    dataset.full_init()
                datasets.append(dataset)
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        # # dk if this is useful
        # num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
        # if num_batch_per_epoch is not None:
        #     world_size = get_world_size()
        #     num_samples = (
        #         num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
        #         world_size)
        #     dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=datasets[0], seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                            **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')
        data_loaders = []
        for dataset in datasets:
            data_loader = DataLoader(
                dataset=dataset,
                sampler=sampler if batch_sampler is None else None,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                worker_init_fn=init_fn,
                **dataloader_cfg)
            data_loaders.append(data_loader)
        return data_loaders
    

@LOOPS.register_module()
class CLValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        self._runner = runner
        self.task_id = 1
        self.dataloader_cfg = dataloader
        self.dataloader = self.build_testcldataloader(dataloader)
        dataloader = self.dataloader
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self.task_id = self.runner.train_loop.task_id
        self.dataloader = self.build_testcldataloader(self.dataloader_cfg)
        # self.index = 1
        # Metrics = dict()
        # for i in range(self.index):
        #     self.dataloader = self.dataloader
        #     for idx, data_batch in enumerate(self.dataloader):
        #         self.run_iter(idx, data_batch)
        #     # compute metrics
        #     metrics = self.evaluator.evaluate(len(self.dataloader.dataset)) # add task id?
        #     avg_v = 0.0
        #     for k,v in metrics.items():
        #         Metrics[f'task_{i+1}_' + k] = v
        #         avg_v += v
        #     avg_v /= float(len(Metrics))
        #     Metrics['task_avg_'+k] = avg_v
        # self.runner.call_hook('after_val_epoch', metrics=Metrics)


        self.dataloader = self.dataloader
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset)) # add task id?

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

    def build_testcldataloader(self, dataloader):
        diff_rank_seed = self.runner._randomness_cfg.get(
            'diff_rank_seed', False)
        seed = self.runner.seed
        if isinstance(dataloader, DataLoader):
                return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            task_num = dataset_cfg['sub_labels']
            class_per_task = int(dataset_cfg['classes']/ task_num)
            org_labels = list(np.array(range(class_per_task*self.task_id)))

            # sub_labels.append(list(map(lambda x:new_order[x],org_labels)))
            dataset_cfg.sub_labels = org_labels
            dataset_cfg.seed = seed
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                            **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')

        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader
    


## CHECKPOINTHOOK
from mmengine.hooks import CheckpointHook
from pathlib import Path
from mmengine.dist import is_main_process


@HOOKS.register_module()
class CLCheckpointHook(CheckpointHook):
    out_dir: str

    priority = 'VERY_LOW'

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.
    from math import inf
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 **kwargs) -> None:
        super().__init__(interval, by_epoch, save_optimizer, save_param_scheduler,
                         out_dir, max_keep_ckpts, save_last, save_best, rule, greater_keys,
                         less_keys, file_client_args, filename_tmpl, backend_args, published_keys,
                         save_begin, **kwargs)
        
    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = f'task_{runner.train_loop.task_id}_'+self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore