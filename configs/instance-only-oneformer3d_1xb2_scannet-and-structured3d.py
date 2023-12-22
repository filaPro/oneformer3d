_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(imports=['oneformer3d'])

# model settings
num_classes_structured3d = 28
num_classes_scannet = 18
voxel_size = 0.05
blocks = 5
num_channels = 64

model = dict(
    type='InstanceOnlyOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    num_classes_1dataset=num_classes_structured3d,
    num_classes_2dataset=num_classes_scannet,
    prefix_1dataset='structured3d', 
    prefix_2dataset ='scannet',
    voxel_size=voxel_size,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(blocks)],
        return_blocks=True),
    decoder=dict(
        type='OneDataQueryDecoder',
        num_layers=3,
        num_queries_1dataset=400, 
        num_queries_2dataset=400,
        num_classes_1dataset=num_classes_structured3d, 
        num_classes_2dataset=num_classes_scannet,
        prefix_1dataset='structured3d', 
        prefix_2dataset ='scannet',
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True),
    criterion=dict(
        type='OneDataCriterion',
        matcher=dict(
            type='HungarianMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='MaskBCECost', weight=1.0),
                dict(type='MaskDiceCost', weight=1.0)]),
        loss_weight=[0.5, 1.0, 1.0, 0.5],
        non_object_weight=0.05,
        num_classes_1dataset=num_classes_structured3d,
        num_classes_2dataset=num_classes_scannet,
        fix_dice_loss_weight=True,
        iter_matcher=True),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=400, 
        score_thr=0.0, 
        npoint_thr=100, 
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear'))

# structured3d dataset settings
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')
dataset_type_structured3d = 'Structured3DSegDataset'
data_root_structured3d = 'data/structured3d/bins'

class_names_structured3d = (
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'picture', 'counter', 'desk', 'shelves', 'curtain', 'dresser',
    'pillow', 'mirror', 'ceiling', 'fridge', 'television', 'night stand',
    'toilet', 'sink', 'lamp', 'bathtub', 'structure', 'furniture', 'prop')
metainfo_structured3d = dict(
    classes=class_names_structured3d,
    ignore_index=num_classes_structured3d)

train_pipeline_structured3d = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample_',
        num_points=200000),
    dict(type='PointSegClassMapping'),
    dict(type='PointInstClassMapping_',
        num_classes=num_classes_structured3d,
        structured3d=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.14, 0.14],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor_', 
         color_mean=[127.5, 127.5, 127.5]),
    dict(type='SkipEmptyScene'),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=-1),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'pts_semantic_mask', 'pts_instance_mask',
            'elastic_coords', 'gt_labels_3d'
        ])
]

# scannet dataset settings
dataset_type_scannet = 'ScanNetDataset'
data_root_scannet = 'data/scannet/'

class_names_scannet = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin')
metainfo_scannet = dict(
    classes=class_names_scannet,
    ignore_index=num_classes_scannet)

train_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='PointSegClassMapping'),
    dict(type='PointInstClassMapping_',
        num_classes=num_classes_scannet),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor_', 
         color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 
            'pts_instance_mask', 'elastic_coords'
        ])
]
test_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor',
                color_mean=[127.5, 127.5, 127.5]),
        ]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[
            dict(
                type=dataset_type_structured3d,
                data_root=data_root_structured3d,
                ann_file='structured3d_infos_train.pkl',
                metainfo=metainfo_structured3d,
                data_prefix=data_prefix,
                pipeline=train_pipeline_structured3d,
                ignore_index=num_classes_structured3d,
                scene_idxs=None,
                test_mode=False),
            dict(
                type='RepeatDataset',
                times=10,
                dataset=dict(
                    type=dataset_type_scannet,
                    data_root=data_root_scannet,
                    ann_file='scannet_oneformer3d_infos_train.pkl',
                    data_prefix=data_prefix,
                    metainfo=metainfo_scannet,
                    pipeline=train_pipeline_scannet,
                    test_mode=False)),
            dict(
                type='RepeatDataset',
                times=10,
                dataset=dict(
                    type=dataset_type_scannet,
                    data_root=data_root_scannet,
                    ann_file='scannet_oneformer3d_infos_val.pkl',
                    data_prefix=data_prefix,
                    metainfo=metainfo_scannet,
                    pipeline=train_pipeline_scannet,
                    test_mode=False))]))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_scannet,
        data_root=data_root_scannet,
        ann_file='scannet_oneformer3d_infos_val.pkl',
        metainfo=metainfo_scannet,
        data_prefix=data_prefix,
        pipeline=test_pipeline_scannet,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='InstanceSegMetric_')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=600000, 
                       power=0.9, by_epoch=False)
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=25000))

train_cfg = dict(
    type='IterBasedTrainLoop',  # Use iter-based training loop
    max_iters=600000,  # Maximum iterations
    val_interval=25000)  # Validation interval
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
