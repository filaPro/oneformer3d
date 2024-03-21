# Adapted from mmdet3d/datasets/transforms/loading.py
import mmengine
import numpy as np

from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.datasets.transforms.loading import get
from mmdet3d.datasets.transforms.loading import NormalizePointsColor
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadAnnotations3D_(LoadAnnotations3D):
    """Just add super point mask loading.
    
    Args:
        with_sp_mask_3d (bool): Whether to load super point maks. 
    """

    def __init__(self, with_sp_mask_3d, **kwargs):
        self.with_sp_mask_3d = with_sp_mask_3d
        super().__init__(**kwargs)

    def _load_sp_pts_3d(self, results):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        sp_pts_mask_path = results['super_pts_path']

        try:
            mask_bytes = get(
                sp_pts_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            sp_pts_mask = np.frombuffer(
                mask_bytes, dtype=np.int64).copy()
        except ConnectionError:
            mmengine.check_file_exist(sp_pts_mask_path)
            sp_pts_mask = np.fromfile(
                sp_pts_mask_path, dtype=np.int64)

        results['sp_pts_mask'] = sp_pts_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_path.split("/")[-1][:-4]
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_sp_mask_3d:
            results = self._load_sp_pts_3d(results)
        return results


@TRANSFORMS.register_module()
class NormalizePointsColor_(NormalizePointsColor):
    """Just add color_std parameter.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
        color_std (list[float]): Std color of the point cloud.
            Default value is from SPFormer preprocessing.
    """

    def __init__(self, color_mean, color_std=127.5):
        self.color_mean = color_mean
        self.color_std = color_std

    def transform(self, input_dict):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.
                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        if self.color_std is not None:
            points.color = points.color / \
                points.color.new_tensor(self.color_std)
        input_dict['points'] = points
        return input_dict
