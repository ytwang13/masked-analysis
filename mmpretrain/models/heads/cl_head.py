# Copyright (c) OpenMMLab. All rights reserved.
from typing import  List, Dict, Union, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, initialize
from mmengine.device import get_device
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .cls_head import ClsHead


@MODELS.register_module()
class incLinearClsHead(ClsHead):
    """Modified Classifier head with several hidden fc layer and a output fc layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence[int]): Number of channels in the hidden fc
            layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mid_channels: Sequence[int],
                 dropout_rate: float = 0.,
                 norm_cfg: Optional[Dict] = dict(type='BN1d'),
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 init_cfg:Optional[Dict] = dict(type='Normal', layer='Linear', std=0.01),
                 loss_weight:Optional[Union[float, Dict]] = 0.,
                 mask_ratio:Optional[float] = 0.5,
                 mask_multi:Optional[int] = 0,
                 mask_loss:Optional[Dict] = None,
                 mask_mode:Optional[Dict] = 's',
                 if_wf:Optional[int] = 0,
                 **kwargs):
        super(incLinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        self.init_cfg = init_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels_old = None
        self.loss_weight = loss_weight
        self.mask_ratio = mask_ratio
        self.mask_loss = None
        self.mask_mode = mask_mode
        self.mask_multi = mask_multi
        self.if_wf = if_wf
        if mask_loss is not None:
            self.mask_loss = MODELS.build(mask_loss) 
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # assert isinstance(mid_channels, Sequence), \
        #     f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
        #     f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()

    def _init_weights(self):
        initialize(self.layers, self.init_cfg)
        if self.mid_channels is not None:
            # _lay = self.layers[-1]
            for i in range(len(self.mid_channels)):
                self.layers[i] = self.old_layers[i]
        weight_old = self.old_layers[-1].fc.weight.data
        bias_old = self.old_layers[-1].fc.bias.data
        self.layers[-1].fc.weight.data[:self.channels_old] = weight_old
        self.layers[-1].fc.bias.data[:self.channels_old] = bias_old
        # self.mask_token = self.oldmask_token
        # self.smask_token = self.oldsmask_token
        del self.old_layers
        # del self.oldsmask_token
        # del self.oldmask_token
        del weight_old
        del bias_old

    def add_head(self, num_class=None):
        if num_class is not None:
            self.channels_old = self.num_classes
            self.num_classes += num_class
        self.old_layers = self.layers
        # self.oldmask_token = self.mask_token
        # self.oldsmask_token = self.smask_token
        self._init_layers()
        self._init_weights()
        self.to(get_device())
        # self.mask_token.to(get_device())
        # self.smask_token.to(get_device())

    def _init_layers(self):
        """"Init layers."""
        # if self.channels_old is None:
        self.mask_token = nn.Parameter(torch.zeros(1, int(self.in_channels * self.mask_ratio)))
        self.smask_token = nn.Parameter(torch.zeros(1, 1))
        self.mask_tokengf = nn.Parameter(torch.zeros(1, int(self.in_channels * self.mask_ratio)), requires_grad=False)
        self.smask_tokengf = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.layers = ModuleList()
        in_channels = self.in_channels
        if self.mid_channels is None:
            self.layers.append(
                LinearBlock(
                    self.in_channels,
                    self.num_classes,
                    dropout_rate=0.,
                    norm_cfg=None,
                    act_cfg=None))
        else:
            for hidden_channels in self.mid_channels:
                self.layers.append(
                    LinearBlock(
                        in_channels,
                        hidden_channels,
                        dropout_rate=self.dropout_rate,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channels = hidden_channels

            self.layers.append(
                LinearBlock(
                    self.mid_channels[-1],
                    self.num_classes,
                    dropout_rate=0.,
                    norm_cfg=None,
                    act_cfg=None))

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage.
        """
        x = feats[-1]
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    @property
    def fc(self):
        """Full connected layer."""
        return self.layers[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score

    def random_masking(
        self,
        x: torch.Tensor,
        mode: str='s',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        x = x[-1]
        if mode =='d':
            self.dropoutmsk = nn.Dropout(self.mask_ratio)
            return tuple([self.dropoutmsk(x)])
        N, D = x.shape  # batch, length, dim
        len_keep = D - int(D * self.mask_ratio)

        noise = torch.rand(N, D, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        if mode =='s':
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, D], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            x_mask = x*(1-mask) + self.smask_token.expand_as(x) *mask
        elif mode =='sgf':
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, D], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            x_mask = x*(1-mask) + self.smask_tokengf.expand_as(x) *mask
        elif mode =='agf':
            ids_keep = ids_shuffle[:, :len_keep]
            x_mask = torch.gather(x, dim=1, index=ids_keep)
            x_mask = torch.cat([x_mask, self.mask_tokengf.expand(size=(x_mask.shape[0],-1))], dim=1)
            x_mask = torch.gather(x_mask, dim=1, index=ids_restore)
        else:
            ids_keep = ids_shuffle[:, :len_keep]
            x_mask = torch.gather(x, dim=1, index=ids_keep)
            x_mask = torch.cat([x_mask, self.mask_token.expand(size=(x_mask.shape[0],-1))], dim=1)
            x_mask = torch.gather(x_mask, dim=1, index=ids_restore)
        return tuple([x_mask])


    def random_maskings(
        self,
        x: torch.Tensor,
        mode: str='s',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        x = x[-1]
        if mode =='d':
            self.dropoutmsk = nn.Dropout(self.mask_ratio)
            return tuple([self.dropoutmsk(x)])
        N, D = x.shape  # batch, length, dim
        len_keep = D - int(D * self.mask_ratio)

        noises = [torch.rand(N, D, device=x.device) for _ in range(self.mask_multi)] # noises in [[0, 1] ...]

        # sort noise for each sample
        ids_shuffles = [torch.argsort(
            noise, dim=1) for noise in noises]  # ascend: small is keep, large is remove
        ids_restores = [torch.argsort(ids_shuffle, dim=1) for ids_shuffle in ids_shuffles]
        
        if mode =='s':
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, D], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            masks = [torch.gather(mask, dim=1, index=ids_restore) for ids_restore in ids_restores]
            x_masks = [x*(1-mask) + self.smask_token.expand_as(x) *mask for mask in masks]
        else:
            ids_keeps = [ids_shuffle[:, :len_keep] for ids_shuffle in ids_shuffles]
            x_masks = [torch.gather(x, dim=1, index=ids_keep) for ids_keep in ids_keeps]
            x_masks = [torch.cat([x_mask, self.mask_token.expand(size=(x_mask.shape[0],-1))], dim=1) for x_mask in x_masks]
            x_masks = [torch.gather(x_mask, dim=1, index=ids_restore) for (x_mask, ids_restore) in zip(x_masks, ids_restores)]
        
        x_mask = torch.cat(x_masks, dim=0)# [BS] -> N*BS,...
        return tuple([x_mask])


    def kd_loss(self, outs, cl_outs, T=2):
        outs = torch.log_softmax(outs / T, dim=1)
        cl_outs = torch.softmax(cl_outs / T, dim=1)
        return -1 * torch.mul(cl_outs, outs).sum() / outs.shape[0]

    def loss(self, 
             cllogits: Tuple[torch.Tensor],
             feats: Tuple[torch.Tensor], 
             data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score[:, self.channels_old:], data_samples, **kwargs)
        return losses
        # def loss
    
    def loss_lwf(self, 
             cllogits: Tuple[torch.Tensor],
             feats: Tuple[torch.Tensor], 
             data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score[:, self.channels_old:], data_samples, **kwargs)
        if self.channels_old is None:
            return losses
        losses['loss_ce'] = losses['loss']
        losses['loss_lwf'] = self.kd_loss(cls_score[:, :self.channels_old], cllogits) * self.loss_weight
        del losses['loss']
        return losses

    def loss_forward(self, 
             cllogits: Tuple[torch.Tensor],
             feats: Tuple[torch.Tensor], 
             data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score[:, self.channels_old:], data_samples, **kwargs)
        return losses


    def loss_msk(self, 
             cllogits: Tuple[torch.Tensor],
             feats: Tuple[torch.Tensor], 
             data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        if self.mask_multi > 0:
            feats_mask = self.random_maskings(feats, mode=self.mask_mode)
        else:
            feats_mask = self.random_masking(feats,mode=self.mask_mode)
        # The part can be traced by torch.fx
        cls_score = self(feats)
        cls_scoremask = self(feats_mask)
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score[:, self.channels_old:], data_samples, **kwargs)
        losses['loss_ce'] = losses['loss']
        del losses['loss']
        if self.channels_old is None:
            return losses
        # losses['loss_mskce'] = self._get_loss(cls_scoremask[:, self.channels_old:], data_samples, **kwargs)['loss']
        ############## lwf_loss
        if self.if_wf:
            losses['loss_lwf'] = self.kd_loss(cls_score[:, :self.channels_old], cllogits) * 10.0

        ############## mask_loss
        if self.mask_multi>0:
            cllogits = torch.cat([cllogits for _ in range(self.mask_multi)], dim=0)###
        if self.mask_loss is not None:
            losses['loss_msk'] = self.mask_loss(cls_scoremask[:, :self.channels_old], cllogits) * self.loss_weight # 0.1
            ## l2 loss weight with lr_mul is useless, lr_mul and mask has the same effect
            # losses['loss_msk'] += self.mask_loss(cls_scoremask[:, self.channels_old:], cls_score[:, self.channels_old:]) * self.loss_weight
        else:
            losses['loss_msk'] = self.kd_loss(cls_scoremask[:, :self.channels_old], cllogits) * self.loss_weight # 5.0
            # kd lrmul 30? so far 3.0 best
        if self.mask_multi>0:
            losses['loss_msk'] /= float(self.mask_multi)
        if self.loss_weight == 0.0:
            del losses['loss_msk']

        return losses
    




class LinearBlock(BaseModule):
    """Linear block for StackedLinearClsHead."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """The forward process."""
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

