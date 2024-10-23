## OneFormer3D: One Transformer for Unified Point Cloud Segmentation

**News**:
 * September, 2024. We release state-of-the-art 3D object detector [UniDet3D](https://github.com/filapro/unidet3d) based on OneFormer3D.
 * :fire: February, 2024. OneFormer3D is now accepted at CVPR 2024.
 * :fire: November, 2023. OneFormer3D achieves state-of-the-art in
   * 3D instance segmentation on ScanNet ([hidden test](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d))
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=oneformer3d-one-transformer-for-unified-point)
     <details>
        <summary>leaderboard screenshot</summary>
        <img src="https://github.com/filaPro/oneformer3d/assets/6030962/e8890fd9-336d-4851-85cb-06fbbb60abe3" alt="ScanNet leaderboard"/>
     </details>
   * 3D instance segmentation on S3DIS (6-Fold)
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=oneformer3d-one-transformer-for-unified-point)
   * 3D panoptic segmentation on ScanNet
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/panoptic-segmentation-on-scannet)](https://paperswithcode.com/sota/panoptic-segmentation-on-scannet?p=oneformer3d-one-transformer-for-unified-point)
   * 3D object detection on ScanNet (w/o TTA)
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=oneformer3d-one-transformer-for-unified-point)
   * 3D semantic segmentation on ScanNet (val, w/o extra training data) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=oneformer3d-one-transformer-for-unified-point)

This repository contains an implementation of OneFormer3D, a 3D (instance, semantic, and panoptic) segmentation method introduced in our paper:

> **OneFormer3D: One Transformer for Unified Point Cloud Segmentation**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ),
> [Danila Rukhovich](https://github.com/filaPro)
> <br>
> Samsung Research<br>
> https://arxiv.org/abs/2311.14405

### Installation

For convenience, we provide a [Dockerfile](Dockerfile).
This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework `v1.1.0`. If installing without docker please follow their [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md).


### Getting Started

Please see [test_train.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/user_guides/train_test.md) for basic usage examples.
For ScanNet and ScanNet200 datasets preprocessing please follow our [instruction](data/scannet). It differs from original mmdetection3d only by adding superpoint clustering. For S3DIS preprocessing we follow original [instruction](https://github.com/open-mmlab/mmdetection3d/tree/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/data/s3dis) from mmdetection3d. We also [support](data/structured3d) Structured3D dataset for pre-training.

Important notes:
 * The metrics from our paper can be achieved in several ways, we just choose the most stable one for each dataset in this repository.
 * If you are interested in only one of three segmentation tasks, it is possible to achieve slightly better metrics, than declared in our paper. Specifically, increasing `model.criterion.sem_criterion.loss_weight` in config file leads to better semantic metrics, and decreasing improve instance metrics.
 * All models can be trained with a single GPU with 32 Gb memory (or even 24 Gb for ScanNet dataset). If you face issues with RAM during instance segmentation evaluation at validation or test stages feel free to decrease `model.test_cfg.topk_insts` in config file.
 * Due to the bug in SpConv we [reshape](tools/fix_spconv_checkpoint.py) backbone weights between train and test stages.

#### ScanNet

For ScanNet we present the model with [SpConv](https://github.com/traveller59/spconv) backbone, superpoint pooling, selecting all queries, and predicting semantics directly from instance queries. Backbone is initialized from [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) checkpoint. It should be [downloaded](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/sstnet_scannet.pth) and put to `work_dirs/tmp` before training.

```shell
# train (with validation)
python tools/train.py configs/oneformer3d_1xb4_scannet.py
# test
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/oneformer3d_1xb4_scannet/epoch_512.pth \
    --out-path work_dirs/oneformer3d_1xb4_scannet/epoch_512.pth
python tools/test.py configs/oneformer3d_1xb4_scannet.py \
    work_dirs/oneformer3d_1xb4_scannet/epoch_512.pth

```

#### ScanNet200

For ScanNet200 we present the model with [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) backbone, superpoint pooling, selecting all queries, and predicting semantics directly from instance queries. Backbone is initialized from [Mask3D](https://github.com/JonasSchult/Mask3D) checkpoint. It should be [downloaded](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/mask3d_scannet200.pth) and put to `work_dirs/tmp` before training.

```shell
# train (with validation)
python tools/train.py configs/oneformer3d_1xb4_scannet200.py
# test
python tools/test.py configs/oneformer3d_1xb4_scannet200.py \
    work_dirs/oneformer3d_1xb4_scannet/epoch_512.pth
```

#### S3DIS

For S3DIS we present the model with [SpConv](https://github.com/traveller59/spconv) backbone, w/o superpoint pooling, w/o query selection, and with separate semantic queries. Backbone is pretrained on Structured3D and ScanNet. It can be [downloaded](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth) and put to `work_dirs/tmp` before training or trained with our code. We train the model on Areas 1, 2, 3, 4, 6 and test on Area 5. To change this split feel free to modify `train_area` and `test_area` parameters in config.

```shell
# pre-train
python tools/train.py configs/instance-only-oneformer3d_1xb2_scannet-and-structured3d.py
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/instance-only-oneformer3d_1xb2_scannet-and-structured3d/iter_600000.pth \
    --out-path work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth
# train (with validation)
python tools/train.py configs/oneformer3d_1xb2_s3dis-area-5.py
# test
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/oneformer3d_1xb2_s3dis-area-5/epoch_512.pth \
    --out-path work_dirs/oneformer3d_1xb2_s3dis-area-5/epoch_512.pth
python tools/test.py configs/oneformer3d_1xb2_s3dis-area-5.py \
    work_dirs/oneformer3d_1xb2_s3dis-area-5/epoch_512.pth
```

### Models

Metric values in the table are given for the provided checkpoints and may vary a little from the ones in our paper. Due to randomness it may be needed to run training with the same config for several times to achieve the best metrics.

| Dataset | mAP<sub>25</sub> | mAP<sub>50</sub> | mAP | mIoU | PQ | Download |
|:-------:|:----------------:|:----------------:|:---:|:----:|:--:|:--------:|
| ScanNet | 86.7 | 78.8 | 59.3 | 76.4 | 70.7 | [model](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet.pth) &#124; [log](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet.log) &#124; [config](configs/oneformer3d_1xb4_scannet.py) |
| ScanNet200 | 44.6 | 40.9 | 30.2 | 29.4 | 29.7 | [model](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet200.pth) &#124; [log](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet200.log) &#124; [config](configs/oneformer3d_1xb4_scannet200.py) |
| S3DIS | 80.6 | 72.7 | 58.0 | 71.9 | 64.6 | [model](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb2_s3dis-area-5.pth) &#124; [log](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/oneformer3d_1xb2_s3dis-area-5.log) &#124; [config](configs/oneformer3d_1xb2_s3dis-area-5.py) |

### Example Predictions

<p align="center">
  <img src="https://github.com/filaPro/oneformer3d/assets/6030962/12809615-7ed5-46a0-9321-747451862295" alt="ScanNet predictions"/>
</p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{kolodiazhnyi2024oneformer3d,
  title={Oneformer3d: One transformer for unified point cloud segmentation},
  author={Kolodiazhnyi, Maxim and Vorontsova, Anna and Konushin, Anton and Rukhovich, Danila},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20943--20953},
  year={2024}
}
```
