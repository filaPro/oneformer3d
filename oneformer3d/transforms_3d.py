import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag


@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        if self.merge_non_stuff_cls:
            gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class PointInstClassMapping_(BaseTransform):
    """Delete instances from non-instaces classes.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Added Keys:
    - gt_labels_3d (int)

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes, structured3d=False):
        self.num_classes = num_classes
        self.structured3d = structured3d

    def transform(self, input_dict):
        """Private function for deleting 
            instances from non-instaces classes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            are updated in the result dict. 'gt_labels_3d' is added.
        """

        # because pts_instance_mask contains instances from non-instaces 
        # classes
        pts_instance_mask = np.array(input_dict['pts_instance_mask'])
        pts_semantic_mask = input_dict['pts_semantic_mask']

        if self.structured3d:
            # wall as one instance
            pts_instance_mask[pts_semantic_mask == 0] = \
                pts_instance_mask.max() + 1
            # floor as one instance
            pts_instance_mask[pts_semantic_mask == 1] = \
                pts_instance_mask.max() + 1
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1

        idxs = np.unique(pts_instance_mask)
        mapping = np.zeros(np.max(idxs) + 2, dtype=int)
        new_idxs = np.arange(len(idxs))
        if idxs[0] == -1:
            mapping[idxs] = new_idxs - 1
            new_idxs = new_idxs[:-1]
        else:
            mapping[idxs] = new_idxs
        pts_instance_mask = mapping[pts_instance_mask]

        input_dict['pts_instance_mask'] = pts_instance_mask
        input_dict['pts_semantic_mask'] = pts_semantic_mask

        gt_labels = np.zeros(len(new_idxs), dtype=int)
        for inst in new_idxs:
            gt_labels[inst] = pts_semantic_mask[pts_instance_mask == inst][0]

        input_dict['gt_labels_3d'] = gt_labels

        return input_dict


@TRANSFORMS.register_module()
class PointSample_(PointSample):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.
        
        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """

        point_range = range(len(points))
        choices = np.random.choice(point_range, 
                                   min(num_samples, len(points)))
        
        return points[choices], choices

    def transform(self, input_dict):
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask',
            'pts_semantic_mask', sp_pts_mask' keys are updated in the 
            result dict.
        """
        points = input_dict['points']
        points, choices = self._points_random_sampling(
            points, self.num_points)
        input_dict['points'] = points
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask
        return input_dict
    
@TRANSFORMS.register_module()
class SkipEmptyScene(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict['gt_labels_3d']) != 0:
            self.inst = input_dict['pts_instance_mask']
            self.sem = input_dict['pts_semantic_mask']
            self.gt_labels = input_dict['gt_labels_3d']
            self.points = input_dict['points']
        else:
            input_dict['pts_instance_mask'] = self.inst
            input_dict['pts_semantic_mask'] = self.sem 
            input_dict['gt_labels_3d'] = self.gt_labels
            input_dict['points'] = self.points

        return input_dict
