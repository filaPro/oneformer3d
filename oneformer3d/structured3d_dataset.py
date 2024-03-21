import numpy as np

from mmengine.dataset.dataset_wrapper import ConcatDataset 
from mmengine.dataset.base_dataset import BaseDataset
from mmdet3d.datasets.seg3d_dataset import Seg3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class Structured3DSegDataset(Seg3DDataset):
    METAINFO = {
        'classes':
        ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
         'window', 'picture', 'counter', 'desk', 'shelves', 'curtain',
         'dresser', 'pillow', 'mirror', 'ceiling', 'fridge', 'television',
         'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'structure',
         'furniture', 'prop'),
        'palette': [[135, 141, 249], [91, 186, 154], [134, 196, 138],
                    [205, 82, 150], [245, 38, 29], [238, 130,249], [189, 22, 4],
                    [128, 94, 103], [121, 74, 63], [98, 252, 9], [227, 8, 226],
                    [224, 58, 233], [244, 26, 146], [50, 62, 237],
                    [141, 30, 106], [60, 187, 63], [206, 106, 254],
                    [164, 85, 194], [187, 218, 244], [244, 140, 56],
                    [118, 8, 242], [88, 60, 134], [230, 110, 157],
                    [174, 48, 170], [3, 119, 80], [69, 148, 166],
                    [171, 16, 47], [81, 66, 251]],
        'seg_valid_class_ids':
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 22, 24, 25,
         32, 33, 34, 35, 36, 38, 39, 40),
        'seg_all_class_ids':
        tuple(range(41)),
    }

    def get_scene_idxs(self, scene_idxs):
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        return np.arange(len(self)).astype(np.int32)


@DATASETS.register_module()
class ConcatDataset_(ConcatDataset):
    """A wrapper of concatenated dataset.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(self,
                 datasets,
                 lazy_init=False,
                 ignore_keys=None):
        self.datasets = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()
