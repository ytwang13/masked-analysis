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
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        self.method_epoch = method_epoch
        self.method_start = method_start
        self.ifop2 = False
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        
    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

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


