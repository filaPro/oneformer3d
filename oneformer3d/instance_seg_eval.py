# Copied from mmdet3d/evaluation/functional/instance_seg_eval.py
# We fix instance seg metric to accept boolean instance seg mask of
# shape (n_points, n_instances) instead of integer mask of shape
# (n_points, ).
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable

from .evaluate_semantic_instance import scannet_eval


# 1) We fix this line: info[file_name]['mask'] = mask[i].
# 2) mask.max() + 1 in for is always equal to 2.
#    We have changed it to mask.shape[0] for iterating over all masks.
def aggregate_predictions(masks, labels, scores, valid_class_ids):
    """Maps predictions to ScanNet evaluator format.

    Args:
        masks (list[torch.Tensor]): Per scene predicted instance masks.
        labels (list[torch.Tensor]): Per scene predicted instance labels.
        scores (list[torch.Tensor]): Per scene predicted instance scores.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[dict]: Per scene aggregated predictions.
    """
    infos = []
    for id, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask = mask.numpy()
        label = label.numpy()
        score = score.numpy()
        info = dict()
        for i in range(mask.shape[0]):
            # match pred_instance['filename'] from assign_instances_for_scan
            file_name = f'{id}_{i}'
            info[file_name] = dict()
            info[file_name]['mask'] = mask[i]
            info[file_name]['label_id'] = valid_class_ids[label[i]]
            info[file_name]['conf'] = score[i]
        infos.append(info)
    return infos


# For some reason the inputs are not torch.Tensor but np.ndarray.
# We just remove torch -> numpy conversion here.
def rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids):
    """Maps gt instance and semantic masks to instance masks for ScanNet
    evaluator.

    Args:
        gt_semantic_masks (list[np.ndarray]): Per scene gt semantic masks.
        gt_instance_masks (list[np.ndarray]): Per scene gt instance masks.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[np.array]: Per scene instance masks.
    """
    renamed_instance_masks = []
    for semantic_mask, instance_mask in zip(gt_semantic_masks,
                                            gt_instance_masks):
        unique = np.unique(instance_mask)
        assert len(unique) < 1000
        for i in unique:
            semantic_instance = semantic_mask[instance_mask == i]
            semantic_unique = np.unique(semantic_instance)
            assert len(semantic_unique) == 1
            if semantic_unique[0] in valid_class_ids:
                instance_mask[instance_mask ==
                              i] = 1000 * semantic_unique[0] + i
        renamed_instance_masks.append(instance_mask)
    return renamed_instance_masks

def instance_seg_eval(gt_semantic_masks,
                      gt_instance_masks,
                      pred_instance_masks,
                      pred_instance_labels,
                      pred_instance_scores,
                      valid_class_ids,
                      class_labels,
                      options=None,
                      logger=None):
    """Instance Segmentation Evaluation.

    Evaluate the result of the instance segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic masks.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance masks.
        pred_instance_masks (list[torch.Tensor]): Predicted instance masks.
        pred_instance_labels (list[torch.Tensor]): Predicted instance labels.
        pred_instance_scores (list[torch.Tensor]): Predicted instance labels.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Names of valid categories.
        options (dict, optional): Additional options. Keys may contain:
            `overlaps`, `min_region_sizes`, `distance_threshes`,
            `distance_confs`. Default: None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(valid_class_ids) == len(class_labels)
    id_to_label = {
        valid_class_ids[i]: class_labels[i]
        for i in range(len(valid_class_ids))
    }
    preds = aggregate_predictions(
        masks=pred_instance_masks,
        labels=pred_instance_labels,
        scores=pred_instance_scores,
        valid_class_ids=valid_class_ids)
    gts = rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids)
    metrics = scannet_eval(
        preds=preds,
        gts=gts,
        options=options,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels,
        id_to_label=id_to_label)
    header = ['classes', 'AP_0.25', 'AP_0.50', 'AP', 'Prec_0.50', 'Rec_0.50']
    rows = []
    for label, data in metrics['classes'].items():
        aps = [data['ap25%'], data['ap50%'], data['ap'], data['prec50%'], data['rec50%']]
        rows.append([label] + [f'{ap:.4f}' for ap in aps])
    aps = metrics['all_ap_25%'], metrics['all_ap_50%'], metrics['all_ap'], metrics['all_prec_50%'], metrics['all_rec_50%']
    footer = ['Overall'] + [f'{ap:.4f}' for ap in aps]
    table = AsciiTable([header] + rows + [footer])
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return metrics
