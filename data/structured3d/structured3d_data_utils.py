import os
import mmengine
import numpy as np
import argparse


class Structured3DData:
    """Structured3DData.

    Args:
        bins_path (str): Root where all bins files are stored.
        point_folder (str): Folder where point bins are stored.
            Defaults to 'points'.
        inst_folder (str): Folder where instance_mask bins are stored.
            Defaults to 'instance_mask'.
        sem_folder (str): Folder where semantic_mask bins are stored.
            Defaults to 'semantic_mask'.
        train_scene_end (str): The last train scene .
            Defaults to 'scene_03000'.
        val_scene_end (str): The last val scene.
            Defaults to 'scene_03250'.
        is_test_needed (bool): Whether or not create test dataset.
            Defaults to True.
    """

    def __init__(self,
                 bins_path,
                 point_folder='points',
                 inst_folder='instance_mask',
                 sem_folder='semantic_mask',
                 bboxs_folder='bboxs',
                 train_scene_end='scene_03000',
                 val_scene_end='scene_03250',
                 is_test_needed=True):
        assert os.path.exists(bins_path)
        points_path = os.path.join(bins_path, point_folder)
        inst_path = os.path.join(bins_path, inst_folder)
        sem_path = os.path.join(bins_path, sem_folder)
        self.bb_path = os.path.join(bins_path, bboxs_folder)
        assert os.path.exists(
            points_path), f'Path to point bins: {points_path} does not exist'
        assert os.path.exists(
            inst_path), f'Path to instance bins: {inst_path} does not exist'
        assert os.path.exists(
            sem_path), f'Path to semantic bins: {sem_path} does not exist'
        assert os.path.exists(
            self.bb_path), f'Path to bboxs npy: {self.bb_path} does not exist'
        self.classes = [
            'unknown', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
            'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
            'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
            'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'fridge',
            'television', 'paper', 'towel', 'shower curtain', 'box',
            'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
            'bathtub', 'bag', 'structure', 'furniture', 'prop'
        ]

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.points = np.array(
            sorted(
                os.listdir(points_path), key=lambda x: int(x.split('_')[1])))
        self.insts = np.array(
            sorted(os.listdir(inst_path), key=lambda x: int(x.split('_')[1])))
        self.sems = np.array(
            sorted(os.listdir(sem_path), key=lambda x: int(x.split('_')[1])))

        if is_test_needed:
            self.train_dataset_points, self.val_dataset_points, \
                self.test_dataset_points = self.get_dataset(
                    self.points, train_scene_end, val_scene_end,
                    is_test_needed)
            self.train_dataset_insts, self.val_dataset_insts, \
                self.test_dataset_insts = self.get_dataset(
                    self.insts, train_scene_end, val_scene_end, is_test_needed)
            self.train_dataset_sems, self.val_dataset_sems, \
                self.test_dataset_sems = self.get_dataset(
                    self.sems, train_scene_end, val_scene_end, is_test_needed)
            self.test_dataset = np.hstack([
                self.test_dataset_points.reshape(-1, 1),
                self.test_dataset_sems.reshape(-1, 1),
                self.test_dataset_insts.reshape(-1, 1)
            ])
        else:
            self.train_dataset_points, self.val_dataset_points = \
                self.get_dataset(
                    self.points, train_scene_end, val_scene_end, is_test_needed)
            self.train_dataset_insts, self.val_dataset_insts = \
                self.get_dataset(
                    self.insts, train_scene_end, val_scene_end, is_test_needed)
            self.train_dataset_sems, self.val_dataset_sems = \
                self.get_dataset(
                    self.sems, train_scene_end, val_scene_end, is_test_needed)

        self.train_dataset = np.hstack([
            self.train_dataset_points.reshape(-1, 1),
            self.train_dataset_sems.reshape(-1, 1),
            self.train_dataset_insts.reshape(-1, 1)
        ])
        self.val_dataset = np.hstack([
            self.val_dataset_points.reshape(-1, 1),
            self.val_dataset_sems.reshape(-1, 1),
            self.val_dataset_insts.reshape(-1, 1)
        ])
        self.datasets = {'train': self.train_dataset, 'val': self.val_dataset}

        if is_test_needed:
            self.datasets['test'] = self.test_dataset

    def __len__(self):
        return len(self.points)

    def get_idx(self, path, train_scene_end, val_scene_end):
        """Get indexes.
        This method gets indexes for train and val datasets.

        Args:
            path (str): Path to the folder with bins.
            train_scene_end (str): The last train scene.
            val_scene_end (str): The last val scene.

        Returns:
            int: Train index
            int: Val index
            
        """
        train_flag = True
        val_flag = True
        for idx, f in enumerate(path):
            if f.startswith(train_scene_end) and train_flag:
                train_idx = idx
                train_flag = False

            if f.startswith(val_scene_end) and val_flag:
                val_idx = idx
                val_flag = False

        return train_idx, val_idx

    def get_dataset(self,
                    path,
                    train_scene_end,
                    val_scene_end,
                    is_test_needed=True):
        """Get datasets
        This method gets train, validation and test if needed datasets

        Args:
            path (str): Path to the folder with bins
            train_scene_end (str): The last train scene 
            val_scene_end (str): The last val scene
            is_test_needed (bool): Whether or not create test dataset
                Defaults to True

        Returns:
            np.ndarray: Train dataset
            np.ndarray: Validtion dataset
            np.ndarray or None: Test dataset
        """
        train_idx, val_idx = self.get_idx(path, train_scene_end, val_scene_end)
        train_dataset = path[:train_idx]
        if is_test_needed:
            val_dataset = path[train_idx:val_idx]
            test_dataset = path[val_idx:]
            return np.array(train_dataset), np.array(val_dataset), \
                np.array(test_dataset)

        else:
            val_dataset = path[train_idx:]
            return np.array(train_dataset), np.array(val_dataset)

    def get_instances(self, sample_idx):
        """Get instances
        This method gets instances for the room

        Args:
            sample_idx (str): Sample_idx of the room

        Returns:
            List[dict]: Instances for the room
        """
        instances = []
        path = os.path.join(self.bb_path, f'{sample_idx}.npy')
        raw_bboxs = np.load(path)
        for i in raw_bboxs:
            bbox = i[:-1].copy()
            if bbox[3] == 0 or bbox[4] == 0 or bbox[5] == 0:
                continue
            bbox[3:] = bbox[3:] * 2
            instances.append({
                'bbox_3d': (bbox).tolist(),
                'bbox_label_3d': int(i[-1])
            })

        return instances

    def get_data_list(self, split='train'):
        """Get data list.
        This method gets data list for the dataset.

        Args:
            split (str): 'train', 'val' or 'test'. Defaults to 'train'.

        Returns:
            List[dict]: Data list for the dataset.
        """
        data_list = []
        dataset = self.datasets[split]
        for f in dataset:
            data_list.append({
                'lidar_points': {
                    'num_pts_feats': 6,
                    'lidar_path': f[0]
                },
                'instances': self.get_instances(f[0].split('.')[0]),
                'pts_semantic_mask_path': f[1],
                'pts_instance_mask_path': f[2],
                'axis_align_matrix': np.eye(4)
            })
        return data_list

    def get_anno(self, split='train'):
        """Get data list.
        This method gets annotations for the dataset.

        Args:
            split (str): 'train', 'val' or 'test'. Defaults to 'train'.

        Returns:
            dict: Annotations for the dataset.
        """
        anno = {
            'metainfo': {
                'categories': self.cat2label,
                'dataset': 'Structured3D',
                'info_version': '1.0'
            }
        }

        anno['data_list'] = self.get_data_list(split)
        return anno


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bins-root',
        required=True,
        help='Enter here the path to the bins folder',
        type=str)
    args = parser.parse_args()
    pkl_prefix = 'structured3d'
    dataset = Structured3DData(args.bins_root)
    train_anno = dataset.get_anno(split='train')
    val_anno = dataset.get_anno(split='val')
    test_anno = dataset.get_anno(split='test')
    filename_train = os.path.join(
        args.bins_root, f'{pkl_prefix}_infos_train.pkl')
    filename_val = os.path.join(
        args.bins_root, f'{pkl_prefix}_infos_val.pkl')
    filename_test = os.path.join(
        args.bins_root, f'{pkl_prefix}_infos_test.pkl')
    mmengine.dump(train_anno, filename_train, 'pkl')
    mmengine.dump(val_anno, filename_val, 'pkl')
    mmengine.dump(test_anno, filename_test, 'pkl')
