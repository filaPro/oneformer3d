## OneFormer3D: One Transformer for Unified Point Cloud Segmentation

**News**:
 * The code and pre-trained models for all datasets will be released soon!
 * :fire: November, 2023. OneFormer3D achieves state-of-the-art in
   * 3D instance segmentation on ScanNet ([hidden test](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d))
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=oneformer3d-one-transformer-for-unified-point)
     <details>
        <summary>leaderboard screenshot</summary>
        <img src="https://github.com/filaPro/oneformer3d/assets/6030962/e8890fd9-336d-4851-85cb-06fbbb60abe3"  alt="ScanNet leaderboard"/>
     </details>
   * 3D semantic segmentation on S3DIS (Area-5)
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-s3dis?p=oneformer3d-one-transformer-for-unified-point)
   * 3D instance segmentation on S3DIS (6-Fold)
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=oneformer3d-one-transformer-for-unified-point)
   * 3D panoptic segmentation on ScanNet
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/panoptic-segmentation-on-scannetv2)](https://paperswithcode.com/sota/panoptic-segmentation-on-scannetv2?p=oneformer3d-one-transformer-for-unified-point)
   * 3D object detection on ScanNet (w/o TTA)
     [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=oneformer3d-one-transformer-for-unified-point)
   * 3D semantic segmentation on ScanNet (val, w/o extra training data) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/oneformer3d-one-transformer-for-unified-point/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=oneformer3d-one-transformer-for-unified-point)

This repository contains an implementation of OneFormer3D, a 3D (instance, semantic, and panoptic) segmentation method introduced in our paper:

> **OneFormer3D: One Transformer for Unified Point Cloud Segmentation**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> [Danila Rukhovich](https://github.com/filaPro)
> <br>
> Samsung Research<br>
> https://arxiv.org/abs/2311.14405

### Example Predictions

<p align="center">
  <img src="https://github.com/filaPro/oneformer3d/assets/6030962/12809615-7ed5-46a0-9321-747451862295" alt="ScanNet predictions"/>
</p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@misc{kolodiazhnyi2023oneformer3d,
  url = {https://arxiv.org/abs/2311.14405},
  author = {Kolodiazhnyi, Maxim and Vorontsova, Anna and Konushin, Anton and Rukhovich, Danila},
  title = {OneFormer3D: One Transformer for Unified Point Cloud Segmentation},
  publisher = {arXiv},
  year = {2023}
}
```
