import os
import cv2
import numpy as np


COLOR_TO_LABEL = {
    (0, 0, 0): 'unknown',
    (174, 199, 232): 'wall',
    (152, 223, 138): 'floor',
    (31, 119, 180): 'cabinet',
    (255, 187, 120): 'bed',
    (188, 189, 34): 'chair',
    (140, 86, 75): 'sofa',
    (255, 152, 150): 'table',
    (214, 39, 40): 'door',
    (197, 176, 213): 'window',
    (148, 103, 189): 'bookshelf',
    (196, 156, 148): 'picture',
    (23, 190, 207): 'counter',
    (178, 76, 76): 'blinds',
    (247, 182, 210): 'desk',
    (66, 188, 102): 'shelves',
    (219, 219, 141): 'curtain',
    (140, 57, 197): 'dresser',
    (202, 185, 52): 'pillow',
    (51, 176, 203): 'mirror',
    (200, 54, 131): 'floor mat',
    (92, 193, 61): 'clothes',
    (78, 71, 183): 'ceiling',
    (172, 114, 82): 'books',
    (255, 127, 14): 'fridge',
    (91, 163, 138): 'television',
    (153, 98, 156): 'paper',
    (140, 153, 101): 'towel',
    (158, 218, 229): 'shower curtain',
    (100, 125, 154): 'box',
    (178, 127, 135): 'whiteboard',
    (120, 185, 128): 'person',
    (146, 111, 194): 'night stand',
    (44, 160, 44): 'toilet',
    (112, 128, 144): 'sink',
    (96, 207, 209): 'lamp',
    (227, 119, 194): 'bathtub',
    (213, 92, 176): 'bag',
    (94, 106, 211): 'structure',
    (82, 84, 163): 'furniture',
    (100, 85, 144): 'prop'
}

colors_and_ids = {k: i for i, (k, s) in enumerate(COLOR_TO_LABEL.items())}
rgbs = np.array(list(colors_and_ids.keys()))
ids = np.array(list(colors_and_ids.values()))
mapping = np.zeros(shape=(256, 256, 256))
mapping[rgbs[:, 0], rgbs[:, 1], rgbs[:, 2]] = ids


class Structured3DScene():
    """Structured3DScene

    Args:
        path_to_scenes (str): Root path to the unziped scenes.
        path_to_bb (str): Root to the unziped bounding boxes
            and annotations data.
        resolution (str): The resolution of the images.
        scene_id (int): Scene index.
    """

    def __init__(self, path_to_scenes, path_to_bb, resolution, scene_id):
        self.resolution = resolution
        self.path_to_bb = path_to_bb
        path = path_to_scenes
        scene_id = f'{scene_id:05d}'
        self.scene_id = scene_id
        self.scene_path = os.path.join(
            path, f'scene_{scene_id}', '2D_rendering')
        room_ids = [p for p in os.listdir(self.scene_path)]
        self.depth_paths = [
            os.path.join(*[
                self.scene_path, room_id, 'panorama', self.resolution,
                'depth.png'
            ]) for room_id in room_ids
        ]

        self.camera_paths = [
            os.path.join(
                *[self.scene_path, room_id, 'panorama', 'camera_xyz.txt'])
            for room_id in room_ids
        ]

        self.rgb_paths = [
            os.path.join(*[
                self.scene_path, room_id, 'panorama', self.resolution,
                'rgb_coldlight.png'
            ]) for room_id in room_ids
        ]

        self.seman_paths = [
            os.path.join(*[
                self.scene_path, room_id, 'panorama', self.resolution,
                'semantic.png'
            ]) for room_id in room_ids
        ]

        self.inst_paths = [
            os.path.join(*[
                self.path_to_bb, f'scene_{self.scene_id}', '2D_rendering',
                room_id, f'panorama/{self.resolution}', 'instance.png'
            ]) for room_id in room_ids
        ]

        self.camera_centers = self.read_camera_center()
        self.point_cloud = self.generate_point_cloud()

    def read_camera_center(self):
        """Read the camera centers.
        This method gets information about camera centers.
        
        Returns:
            List[np.ndarray]: camera centers for every room in the scene.
        """
        camera_centers = []
        for i in range(len(self.camera_paths)):
            if os.path.exists(self.camera_paths[i]):
                with open(self.camera_paths[i], 'r') as f:
                    line = f.readline()
                center = list(map(float, line.strip().split(' ')))
                camera_centers.append(
                    np.asarray([center[0], center[1], center[2]]))

        return camera_centers

    def generate_point_cloud(self):
        """Generate data.
        This method gets point_clouds, semantics, instances
        and bboxs for every room in the scene.

        Returns:
            dict: Processed point_clouds, semantics, instances, bboxs.
        """
        points = {}
        labels = []
        point_clouds = []
        insts = []
        bboxs = []
        for i in range(len(self.depth_paths)):
            try:
                depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_ANYDEPTH)
                # ------------------- #
                H, W = depth.shape
                x_tick = 180.0 / H
                y_tick = 360.0 / W
                x = np.arange(H)
                y = np.arange(W)
                x = np.broadcast_to(x.reshape(-1, 1), (H, W))
                y = np.broadcast_to(y.reshape(-1), (H, W))
                alpha = 90 - (x * x_tick)
                beta = y * y_tick - 180
                xy_offset = depth * np.cos(np.deg2rad(alpha))
                x_offset = xy_offset * np.sin(
                    np.deg2rad(beta)) + self.camera_centers[i][0]
                y_offset = xy_offset * np.cos(
                    np.deg2rad(beta)) + self.camera_centers[i][1]
                z_offset = depth * np.sin(
                    np.deg2rad(alpha)) + self.camera_centers[i][2]
                temp = np.hstack([
                    x_offset.reshape(-1, 1),
                    y_offset.reshape(-1, 1),
                    z_offset.reshape(-1, 1)
                ]) / 1000
                # ------------------- #
                # Read RGB image
                rgb_img = cv2.imread(self.rgb_paths[i]).reshape(-1, 3)
            
                # ------------------- #
                # Read semantic image
                semantic = cv2.imread(self.seman_paths[i])
                semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
                # ------------------- #
                # Read instance image
                inst = cv2.imread(self.inst_paths[i], cv2.IMREAD_UNCHANGED)
            except:
                continue

            semantic = semantic.reshape(-1, 3)
            cur_labels = mapping[
                semantic[:, 0], semantic[:, 1], semantic[:, 2]].copy()
            inst = inst.reshape(-1)
            inst = np.where(inst == 65535, -1, inst)

            if np.unique(inst)[0] == -1:
                instance_unique = np.unique(inst)[1:]
            else:
                instance_unique = np.unique(inst)
            if temp.shape[0] != inst.shape[0]:
                print(
                    f'Error - point_cloud shape {temp.shape[0]} '
                    f'!= inst.shape {inst.shape[0]}')
                continue

            for inst_id in instance_unique:
                cur_labels[inst == inst_id] = \
                    np.unique(cur_labels[inst == inst_id])[0]

            inst[cur_labels == 1] = -1
            inst[cur_labels == 2] = -1

            if np.unique(inst)[0] == -1:
                instance_unique = np.unique(inst)[1:]
            else:
                instance_unique = np.unique(inst)
            
            for inst_id in instance_unique:
                assert len(np.unique(cur_labels[inst == inst_id])) == 1

            if len(inst[cur_labels == 1]) != 0:
                assert len(np.unique(inst[cur_labels == 1])) == 1
                assert np.unique(inst[cur_labels == 1])[0] == -1
            
            if len(inst[cur_labels == 2]) != 0:
                assert len(np.unique(inst[cur_labels == 2])) == 1
                assert np.unique(inst[cur_labels == 2])[0] == -1

            temp_bb = []
            for inst_id in instance_unique:
                indexes = inst == inst_id
                current_points = temp[indexes]
                current_points_min = current_points.min(0)
                current_points_max = current_points.max(0)
                current_points_avg = (
                    current_points_max + current_points_min) / 2
                lwh = (current_points_max - current_points_avg).copy()
                vals, occurs = np.unique(
                    cur_labels[indexes], return_counts=True)
                bbox_labels = vals[occurs.argmax()].copy()
                temp_bb.append(
                    np.hstack([current_points_avg, lwh, bbox_labels]))

            insts.append(inst.copy())
            labels.append(cur_labels)
            point_clouds.append(np.hstack([temp, rgb_img]).copy())
            bboxs.append(temp_bb)

        points['labels'] = labels
        points['point_clouds'] = point_clouds
        points['instances'] = insts
        points['bboxs'] = bboxs

        return points
