from typing import List, Optional

import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .image import CLImageClassifier
from typing import Dict, Optional, Tuple, Union

@MODELS.register_module()
class LWFcls(CLImageClassifier):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(backbone, neck, head, pretrained,
                         train_cfg, data_preprocessor, init_cfg)
        
    def _run_lwf(self,                
            inputs: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None,
            mode: str = 'tensor'):
        # if self.task
        if self.is_old:
            old_logits = self.fc_old(inputs)
            return self.loss(old_logits, inputs, data_samples)
        return self.loss(None, inputs, data_samples)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,
                   mode: Optional[str]='ce') -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        data = self.data_preprocessor(data, True)
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            losses = self._run_lwf(**data, mode='loss')         
            parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            optim_wrapper.update_params(parsed_losses)
        ###### Multiple
        # optim_wrapper_backbone, optim_wrapper_head = optim_wrapper['backbone'],optim_wrapper['head']
        # with optim_wrapper_backbone.optim_context(self.backbone):
        #     losses = self._run_lwf(data, mode='loss')  # type: ignore
        #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        #     optim_wrapper_backbone.update_params(parsed_losses)
        # with optim_wrapper_head.optim_context(self.head):
        #     losses = self._run_lwf(data, mode='loss')  # type: ignore
        #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        #     optim_wrapper_head.update_params(parsed_losses)
        return log_vars

    def loss(self, cllogits: torch.Tensor, 
             inputs: torch.Tensor,
             data_samples: List[DataSample],) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss_lwf(cllogits, feats, data_samples)