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



@LOOPS.register_module()
class resEpochTrainLoop(EpochBasedTrainLoop):
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
            method_epoch: int = 2,
            method_start: int=1,
            if_lp:Optional = False,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        self.method_epoch = method_epoch
        self.method_start = method_start
        self.ifop2 = False
        self.if_lp = if_lp
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        
    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        while self._epoch < self._max_epochs and not self.stop_training:
            if (self.runner.val_loop is not None
                    and (self._epoch + 1) >= self.val_begin
                    and (self._epoch + 1) % self.val_interval == 0):
                    if self.if_lp:
                        self.lp_configure() ######## later try lp mode (test/train lp-training)
            if (self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0
                    and self.if_lp):
                        self.lp_delconfigure()

            self.run_epoch() # here epoch +1
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                    self.runner.val_loop.run()## here pass self._Epoch
            if self._epoch % self.method_epoch == 0 and self._epoch >self.method_start and self._epoch<self._max_epochs:
                self.runner.logger.info(f'🛫🛫🛫🛫🛫🛫🛫🛫 Now add method at epoch_{self._epoch} 😊😊😊😊😊😊😊😊') 
                self.res_configure()

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
            data_batch, optim_wrapper=self.runner.optim_wrapper)#, mode='lwf'

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
            self.run_iter(idx, data_batch)
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
    
    def res_configure(self):
        ###### add here for LP configuration for last epoch.
        ######################## prepare method init
        if hasattr(self.runner.model, 'module'):
            self.runner.model.module.method_init()
            # self.runner.model.module.head.add_head(self.num_class)
        else:
            self.runner.model.method_init()
        # change to ratioed method
        if not self.ifop2:
            optim_cfg = self.runner.cfg.get('optim_wrapper2') #
            self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)# can this achive the ratio thing?
            self.ifop2 = True
        # self.runner.param_schedulers = self.runner.build_param_scheduler(self.runner.cfg.get('param_scheduler2'))

    def lp_configure(self): # this should be after res_configuration
        if hasattr(self.runner.model, 'module'):
            self.runner.model.module.head._add_lp()
            # self.runner.model.module.head.add_head(self.num_class)
        else:
            self.runner.model.head._add_lp()
        if not self.ifop2:
            optim_cfg = self.runner.cfg.get('optim_wrapperlp') #
            self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)# can this achive the ratio thing?
        else:
            optim_cfg = self.runner.cfg.get('optim_wrapper2lp') #
            self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)# can this achive the ratio thing?

    def lp_delconfigure(self):
        if hasattr(self.runner.model, 'module'):
            self.runner.model.module.head._del_lp()
            # self.runner.model.module.head.add_head(self.num_class)
        else:
            self.runner.model.head._del_lp()
        if not self.ifop2:
            optim_cfg = self.runner.cfg.get('optim_wrapper') #
            self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)# can this achive the ratio thing?
        else:
            optim_cfg = self.runner.cfg.get('optim_wrapper2') #
            self.runner.optim_wrapper = self.runner.build_optim_wrapper(optim_cfg)# can this achive the ratio thing?



@LOOPS.register_module()
class v1ValLoop(BaseLoop):
    """Loop for validation two separate datasets

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
                 fp16: bool = False,
                 if_lp: bool = False) -> None:
        self._runner = runner
        self.task_id = 1
        self.dataloader_cfg = dataloader
        self.dataloaders = self.build_testdataloaders(dataloader)
        self.if_lp = if_lp
        dataloader = self.dataloaders[0]
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
        import copy
        if self.if_lp:
            self.evaluatorlp = runner.build_evaluator(evaluator)
        else:
            self.evaluatorlp = None
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
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

        Metrics = dict()
        for task_id, dataloader in enumerate(self.dataloaders):
            for idx, data_batch in enumerate(dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(dataloader.dataset)) # add task id?
            if task_id >0:
                for key, value in metrics.items():
                    Metrics.update({str(task_id) + key: value})
            else:
                for key, value in metrics.items():
                    Metrics.update({key: value})
            if self.if_lp:
                metricslp = self.evaluatorlp.evaluate(len(dataloader.dataset)) # add task id?
                if task_id >0:
                    for key, value in metricslp.items():
                        Metrics.update({str(task_id) + key +'lp': value})
                else:
                    for key, value in metricslp.items():
                        Metrics.update({key+'lp': value})

        self.runner.call_hook('after_val_epoch', metrics=Metrics)
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
            outputslp = self.runner.model.lp_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        if self.if_lp:
            self.evaluatorlp.process(data_samples=outputslp, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

    def build_testdataloaders(self, dataloader):
        diff_rank_seed = self.runner._randomness_cfg.get(
            'diff_rank_seed', False)
        seed = self.runner.seed
        if isinstance(dataloader, DataLoader):
                return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        datasets = []
        if isinstance(dataset_cfg, list):
            task_num = len(dataset_cfg)
            for task_id in range(task_num):
                dataset = DATASETS.build(dataset_cfg[task_id])
                if hasattr(dataset, 'full_init'):
                    dataset.full_init()
                datasets.append(dataset)
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg
            # dataset = DATASETS.build(dataset)
            # if hasattr(dataset, 'full_init'):
            #     dataset.full_init()

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        samplers = []
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            for task_id in range(len(datasets)):
                sampler = DATA_SAMPLERS.build(
                    sampler_cfg,
                    default_args=dict(dataset=datasets[task_id], seed=sampler_seed))
                samplers.append(sampler)
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_samplers = []
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            for task_id in range(len(samplers)):
                batch_sampler = DATA_SAMPLERS.build(
                    batch_sampler_cfg,
                    default_args=dict(
                        sampler=sampler,
                        batch_size=dataloader_cfg.pop('batch_size')))
                batch_samplers.append(batch_sampler)
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
        for task_id, (dataset, sampler) in enumerate(zip(datasets, samplers)):
            data_loader = DataLoader(
                dataset=dataset,
                sampler=sampler if batch_sampler is None else None,
                batch_sampler=None,
                collate_fn=collate_fn,
                worker_init_fn=init_fn,
                **dataloader_cfg)
            data_loaders.append(data_loader)
        return data_loaders
