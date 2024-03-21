_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 64
num_instance_classes = 13
num_semantic_classes = 13

model = dict(
    type='S3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.05,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=400,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='S3DISUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=5.0),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=450,
        inst_score_thr=0.0,
        pan_score_thr=0.4,
        npoint_thr=300,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0, 1, 2, 3, 4, 5, 6, 12],
        thing_cls=[7, 8, 9, 10, 11]))

# dataset settings
dataset_type = 'S3DISSegDataset_'
data_root = 'data/s3dis/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_area = [1, 2, 3, 4, 6]
test_area = 5

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample_',
        num_points=180000),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
]
test_pipeline = [
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
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

# run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
            type='ConcatDataset',
            datasets=([
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f's3dis_infos_Area_{i}.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    data_prefix=data_prefix,
                    box_type_3d='Depth',
                    backend_args=None) for i in train_area])))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='Depth',
        backend_args=None))
test_dataloader = val_dataloader

class_names = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='S3DIS')
sem_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0, 1, 2, 3, 4, 5, 6, 12],
    thing_class_inds=[7, 8, 9, 10, 11],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=16,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

load_from = 'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=512, val_interval=16)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
