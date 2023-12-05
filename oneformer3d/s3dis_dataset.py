from mmdet3d.registry import DATASETS
from mmdet3d.datasets.s3dis_dataset import S3DISSegDataset


@DATASETS.register_module()
class S3DISSegDataset_(S3DISSegDataset):
    pass