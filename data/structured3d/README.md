## Prepare Structured3D Data for 3D Indoor Instance Segmentation

1. Download panorama zip files and 3d bounding box and annotations zip file to separate folders from the official [Structured3D](https://github.com/bertjiazheng/Structured3D).

After this step the data are expected to be in the following structure:

```
panorama_folder
├── Structured3D_panorama_xx.zip
bb_folder
├── Structured3D_bbox.zip
```

Unzip data by running our script:

```
python unzip.py --panorama-root panorama_folder --output-panorama-root panorama_folder_unziped --bb-root bb_folder --output-bb-root bb_folder_unziped
```

After this step you have the following file structure here:
```
panorama_folder_unziped
├── xxxxxxxx
│   ├── Structured3D
│   │   ├── scene_xxxxx
bb_folder_unziped
├── Structured3D
│   ├── scene_xxxxx
```
2. Preprocess data for offline benchmark by running our script:

```
python data_prepare.py --panorama-root panorama_folder_unziped --bb-root bb_folder_unziped/Structured3D/ --bins-root bins

```
After this step you have the following file structure here:
```
bins
├── bboxs
│   ├── scene_xxxxx_xx.npy
├── instance_mask
│   ├── scene_xxxxx_xx.bin
├── points
│   ├── scene_xxxxx_xx.bin
├── semantic_mask
│   ├── scene_xxxxx_xx.bin
```

3. Generate final pkl data by running:

```
python structured3d_data_utils.py --bins-root bins
```
Overall you achieve the following file structure in `bins` directory:
```
bins
├── bboxs
│   ├── scene_xxxxx_xx.npy
├── instance_mask
│   ├── scene_xxxxx_xx.bin
├── points
│   ├── scene_xxxxx_xx.bin
├── semantic_mask
│   ├── scene_xxxxx_xx.bin
├── structured3d_infos_train.pkl
├── structured3d_infos_val.pkl
├── structured3d_infos_test.pkl
```

