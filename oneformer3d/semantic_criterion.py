import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class ScanNetSemanticCriterion:
    """Semantic criterion for ScanNet.

    Args:
        ignore_index (int): Ignore index.
        loss_weight (float): Loss weight.
    """

    def __init__(self, ignore_index, loss_weight):
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `sem_preds`
                of len batch_size, each of shape
                (n_queries_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic loss value.
        """
        losses = []
        for pred_mask, gt_mask in zip(pred['sem_preds'], insts):
            if self.ignore_index >= 0:
                pred_mask = pred_mask[:, :-1]
            losses.append(F.cross_entropy(
                pred_mask,
                gt_mask.sp_masks.float().argmax(0),
                ignore_index=self.ignore_index))
        loss = self.loss_weight * torch.mean(torch.stack(losses))
        return dict(seg_loss=loss)


@MODELS.register_module()
class S3DISSemanticCriterion:
    """Semantic criterion for S3DIS.

    Args:
        loss_weight (float): loss weight.
        seg_loss (ConfigDict): loss config.
    """

    def __init__(self,
                 loss_weight,
                 seg_loss=dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=True)):
        self.seg_loss = MODELS.build(seg_loss)
        self.loss_weight = loss_weight

    def get_layer_loss(self, layer, aux_outputs, insts):
        """Calculate loss at intermediate level.

        Args:
            layer (int): transformer layer number
            aux_outputs (dict): Predictions with List `masks`
                of len batch_size, each of shape
                (n_points_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        pred_masks = aux_outputs['masks']
        seg_losses = []
        for pred_mask, gt_mask in zip(pred_masks, insts):
            seg_loss = self.seg_loss(
                pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            seg_losses.append(seg_loss)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        return {f'layer_{layer}_seg_loss': seg_loss}

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `masks`
                of len batch_size, each of shape
                (n_points_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        pred_masks = pred['masks']
        seg_losses = []
        for pred_mask, gt_mask in zip(pred_masks, insts):
            seg_loss = self.seg_loss(
                pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            seg_losses.append(seg_loss)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        loss = {'last_layer_seg_loss': seg_loss}

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i = self.get_layer_loss(i, aux_outputs, insts)
                loss.update(loss_i)

        return loss
