import argparse
import os
import numpy as np
from utils import Structured3DScene


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--panorama-root',
        required=True,
        help='Folder with panorama scenes',
        type=str)
    parser.add_argument(
        '--bb-root',
        required=True,
        help='Folder with bb scenes',
        type=str)
    parser.add_argument(
        '--bins-root',
        required=True,
        help='Folder with bin files',
        type=str)

    args = parser.parse_args()
    if not os.path.exists(args.bins_root):
        os.mkdir(args.bins_root)

    inst_bins_path = os.path.join(args.bins_root, 'instance_mask')
    pc_bins_path = os.path.join(args.bins_root, 'points')
    sem_bins_path = os.path.join(args.bins_root, 'semantic_mask')
    bb_bins_path = os.path.join(args.bins_root, 'bboxs')

    if not os.path.exists(inst_bins_path):
        os.mkdir(inst_bins_path)

    if not os.path.exists(pc_bins_path):
        os.mkdir(pc_bins_path)

    if not os.path.exists(sem_bins_path):
        os.mkdir(sem_bins_path)

    if not os.path.exists(bb_bins_path):
        os.mkdir(bb_bins_path)

    sorted_scene_folder = sorted(
        os.listdir(args.panorama_root), key=lambda x: int(x))

    for scenes in sorted_scene_folder:
        path_to_scenes = os.path.join(
            args.panorama_root, scenes, 'Structured3D')
        scenes = sorted(os.listdir(path_to_scenes), key=lambda x: int(x[-5:]))
        for scene in scenes:
            scene_id = int(scene[-5:])
            data = Structured3DScene(
                path_to_scenes, args.bb_root, 'full', scene_id)
            room_nums = len(data.point_cloud['point_clouds'])
            for idx in range(room_nums):
                data.point_cloud['point_clouds'][idx].astype(
                    np.float32).tofile(
                        os.path.join(pc_bins_path, f'{scene}_{idx}.bin'))

                data.point_cloud['labels'][idx].astype(np.int64).tofile(
                    os.path.join(sem_bins_path, f'{scene}_{idx}.bin'))

                data.point_cloud['instances'][idx].astype(np.int64).tofile(
                    os.path.join(inst_bins_path, f'{scene}_{idx}.bin'))

                np.save(
                    os.path.join(bb_bins_path, f'{scene}_{idx}.npy'),
                    data.point_cloud['bboxs'][idx])

            print(f'{scene} is processed')
