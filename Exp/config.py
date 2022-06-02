optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=999999)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'F:\downloads\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'F:\source\repos\Daten\gammher2'
dataset_type = 'CocoDataset'
train_pipeline = [
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    persistent_workers=False,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=
            'F:\source\repos\Daten\gammher2\annotations/instances_train2017.json',
            img_prefix='F:\source\repos\Daten\gammher2\train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
            classes=('0', )),
        pipeline=[
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=
        'F:\source\repos\Daten\gammher2\annotations/instances_val2017.json',
        img_prefix='F:\source\repos\Daten\gammher2\val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('0', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        'F:\source\repos\Daten\gammher2\annotations/instances_val2017.json',
        img_prefix='F:\source\repos\Daten\gammher2\val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('0', )))
max_epochs = 300
num_last_epochs = 15
interval = 10
evaluation = dict(
    save_best='bbox_mAP',
    interval=1,
    dynamic_intervals=[(285, 1)],
    metric='bbox')
seed = 1234
gpu_ids = [0]
work_dir = 'F:\source\repos\InferenceDL\Exp'
total_epochs = 300
device = 'cuda'
