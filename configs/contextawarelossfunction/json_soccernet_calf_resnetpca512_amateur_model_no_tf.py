work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model"
cut_classes = ["Kick-off", "Goal"]
classes=cut_classes
data_root = "/workspace/datasets/amateur-dataset/"

dataset = dict(
    input_fps=25,
    extract_fps=2,
    train=dict(
        type="FeatureClipChunksfromJson",
        path="/workspace/datasets/amateur-dataset/train/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=1500, # prior 6000
        classes=cut_classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        ),
    ),
    valid=dict(
        type="FeatureClipChunksfromJson",
        path="/workspace/datasets/amateur-dataset/valid/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=1500, # prior 6000
        classes=cut_classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        ),
    ),
    test=dict(
        type="FeatureVideosChunksfromJson",
        path="/workspace/datasets/amateur-dataset/test/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=1500, # prior 6000
        classes=cut_classes,
        metric="loose",
        results="results_spotting_test",
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        ),
    ),
)

model = dict(
    type='ContextAware',
    load_weights=None,
    backbone=dict(
        type='PreExtactedFeatures',
        encoder='ResNET_TF2_PCA512',
        feature_dim=512,
        output_dim=512,
        framerate=2),
    neck=dict(
        type='CNN++',
        input_size=512,
        num_classes=2,
        chunk_size=120,
        dim_capsule=16,
        receptive_field=40,
        num_detections=15,
        framerate=2),
    head=dict(
        type='SpottingCALF',
        num_classes=2,
        dim_capsule=16,
        num_detections=15,
        num_layers=2,
        chunk_size=120),
)

runner = dict(
    type="runner_JSON"
)


log_level = "DEBUG"  # The level of logging

training = dict(
    type="trainer_CALF",
    max_epochs=1000,
    evaluation_frequency=1000,
    framerate=2,
    batch_size=32,
    GPU=0,
    criterion = dict(
        type="Combined2x",
        #w_1 = 0.000367,
        w_1=0.5,
        loss_1 = dict(
            type="ContextAwareLoss",
            #K=[[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18],
            #[-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9],
            #[50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9],
            #[100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]],
            #K=[[-100, -100], [-50, -50], [50, 50], [100, 100]],
            K = [[-10,  -100], [-5,-50], [5, 50], [ 10, 100]],
            framerate=2,
            pos_radius=4,
            neg_radius=10,
            hit_radius = 0.4,
            miss_radius = 1.2
        ),
        w_2 = 1.0,
        loss_2 = dict(
            type="SpottingLoss",
            lambda_coord=5.0,
            lambda_noobj=0.5
        ),
    ),
    optimizer = dict(
        type="Adam",
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-07,
        weight_decay=0,
        amsgrad=False
    ),
    scheduler=dict(
        type="ReduceLROnPlateau",
        mode="min",
        LR=1e-3,
        LRe=1e-06,
        patience=25,
        verbose=True,
    ),
)
