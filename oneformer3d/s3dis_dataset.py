from mmdet3d.registry import DATASETS
from mmdet3d.datasets.s3dis_dataset import S3DISDataset


@DATASETS.register_module()
class S3DISSegDataset_(S3DISDataset):
    METAINFO = {
        'classes':
        ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
         'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids':
        tuple(range(13)),
        'seg_all_class_ids':
        tuple(range(14))  # possibly with 'stair' class
    }
