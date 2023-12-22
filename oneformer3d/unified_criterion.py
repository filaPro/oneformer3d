from mmdet3d.registry import MODELS
from .structures import InstanceData_


@MODELS.register_module()
class ScanNetUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
    
    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes

        for i in range(len(pred['masks'])):
            sem_gt = InstanceData_()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n - 1:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n - 1:, :]
            sem_gts.append(sem_gt)

            inst_gt = InstanceData_()
            inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :]
            inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n - 1, :]
            inst_gts.append(inst_gt)
        
        loss = self.inst_criterion(pred, inst_gts)
        loss.update(self.sem_criterion(pred, sem_gts))
        return loss

@MODELS.register_module()
class S3DISUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `sp_inst_masks` of shape 
                        (n_gts_i, n_points_i)
                    `sp_sem_masks` of shape 
                        (n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,).

        Returns:
            Dict: with semantic and instance loss values.
        """
        pred_masks = pred['masks']
        pred_cls = pred['cls_preds']
        pred_scores = pred['scores']
        
        sem_preds = []
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes
        for i in range(len(pred_masks)):
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_cls[i] = pred_cls[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]
            
            sem_gt = InstanceData_()
            inst_gt = InstanceData_()
            sem_gt.sp_masks = insts[i].sp_sem_masks
            sem_gts.append(sem_gt)
            inst_gt.sp_masks = insts[i].sp_inst_masks
            inst_gt.labels_3d = insts[i].labels_3d
            inst_gts.append(inst_gt)

        if 'aux_outputs' in pred:
            sem_aux_outputs = []
            for aux_outputs in pred['aux_outputs']:
                sem_aux_outputs.append(self.prepare_aux_outputs(aux_outputs))
        
        loss = self.inst_criterion(pred, inst_gts)
        loss.update(self.sem_criterion(
            {'masks': sem_preds, 'aux_outputs': sem_aux_outputs}, sem_gts))
        return loss

    def prepare_aux_outputs(self, aux_outputs):
        """Prepare aux outputs for intermediate layers.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points).

        Returns:
            Dict: with semantic predictions.
        """
        pred_masks = aux_outputs['masks']
        pred_cls = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        
        sem_preds = []
        n = self.num_semantic_classes
        for i in range(len(pred_masks)):
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_cls[i] = pred_cls[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]

        return {'masks': sem_preds}
