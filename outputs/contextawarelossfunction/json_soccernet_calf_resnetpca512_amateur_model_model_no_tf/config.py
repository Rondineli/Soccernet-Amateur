classes = [
    'Goal',
    'Kick-off',
]
contextaware_cfg = dict(
    lambda_neg=0.25,
    lambda_pos=2.0,
    lambda_reg=0.5,
    neg_radius=9,
    normalize=True,
    pos_radius=3)
data_root = '/workspace/datasets/amateur-dataset/'
dataset = dict(
    extract_fps=2,
    input_fps=25,
    test=dict(
        chunk_size=120,
        chunks_per_epoch=6000,
        classes=[
            'Goal',
            'Kick-off',
        ],
        data_root='/workspace/datasets/amateur-dataset/',
        dataloader=dict(
            batch_size=1, num_workers=1, pin_memory=True, shuffle=False),
        extract_fps=2,
        framerate=1,
        input_fps=25,
        metric='loose',
        path='/datasets/amateur/test_amateur_annotations.json',
        receptive_field=40,
        results='results_spotting_test',
        type='FeatureVideosChunksfromJson'),
    train=dict(
        chunk_size=120,
        chunks_per_epoch=6000,
        classes=[
            'Goal',
            'Kick-off',
        ],
        data_root='/workspace/datasets/amateur-dataset/',
        dataloader=dict(
            batch_size=256, num_workers=4, pin_memory=True, shuffle=True),
        evaluation_frequency=20,
        extract_fps=2,
        framerate=1,
        input_fps=25,
        path='/datasets/amateur/train_amateur_annotations.json',
        receptive_field=40,
        type='FeatureClipChunksfromJson'),
    valid=dict(
        chunk_size=120,
        chunks_per_epoch=6000,
        classes=[
            'Goal',
            'Kick-off',
        ],
        data_root='/workspace/datasets/amateur-dataset/',
        dataloader=dict(
            batch_size=256, num_workers=4, pin_memory=True, shuffle=True),
        extract_fps=2,
        framerate=1,
        input_fps=25,
        path='/datasets/amateur/valid_amateur_annotations.json',
        receptive_field=40,
        type='FeatureClipChunksfromJson'))
evaluation_frequency = 20
log_level = 'INFO'
model = dict(
    backbone=dict(
        encoder='ResNET_TF2_PCA512',
        feature_dim=512,
        framerate=1,
        output_dim=512,
        type='PreExtactedFeatures'),
    head=dict(
        chunk_size=120,
        dim_capsule=16,
        num_classes=2,
        num_detections=15,
        num_layers=2,
        type='SpottingCALF'),
    load_weights=None,
    neck=dict(
        chunk_size=120,
        dim_capsule=16,
        framerate=1,
        input_size=512,
        num_classes=2,
        num_detections=15,
        receptive_field=40,
        type='CNN++'),
    type='ContextAware')
optimizer = dict(lr=0.0001)
runner = dict(type='runner_JSON')
scheduler = dict(patience=10, type='ReduceLROnPlateau')
training = dict(
    GPU=0,
    batch_size=64,
    criterion=dict(
        loss_1=dict(
            K=[
                [
                    -100,
                    -100,
                ],
                [
                    -50,
                    -50,
                ],
                [
                    50,
                    50,
                ],
                [
                    100,
                    100,
                ],
            ],
            framerate=2,
            hit_radius=0.2,
            miss_radius=0.8,
            neg_radius=20,
            pos_radius=8,
            type='ContextAwareLoss'),
        loss_2=dict(lambda_coord=5.0, lambda_noobj=0.5, type='SpottingLoss'),
        type='Combined2x',
        w_1=1.0,
        w_2=1.0),
    evaluation_frequency=1000,
    framerate=2,
    max_epochs=10,
    optimizer=dict(
        amsgrad=False,
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-07,
        lr=0.001,
        type='Adam',
        weight_decay=0),
    scheduler=dict(
        LR=0.001,
        LRe=1e-06,
        mode='min',
        patience=25,
        type='ReduceLROnPlateau',
        verbose=True),
    type='trainer_CALF')
work_dir = 'outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_model_no_tf'
