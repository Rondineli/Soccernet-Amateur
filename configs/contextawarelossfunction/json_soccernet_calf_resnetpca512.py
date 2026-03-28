"""
CALF model from OSL-Action - source: https://github.com/OpenSportsLab/OSL-ActionSpotting/blob/main/configs/contextawarelossfunction/json_soccernet_calf_resnetpca512.py
Adapted to load only 2 label classes: Kick-off and Goal
"""
_base_ = [
    "../_base_/datasets/json/features_clips_CALF.py",  # dataset config
    "../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py",  # trainer config
]

work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512"

dataset = dict(
    train=dict(
        path=[
            "/workspace/datasets/spotting-OSL/ResNET_PCA512/train/annotations.json"
        ],
        data_root=["/home/ybenzakour/datasets/SoccerNet/"],
        classes = ["Kick-off", "Goal"]
    ),
    valid=dict(
        path="/workspace/datasets/spotting-OSL/ResNET_PCA512/valid/annotations.json",
        classes = ["Kick-off", "Goal"]
    ),
    test=dict(
        path="/workspace/datasets/spotting-OSL/ResNET_PCA512/test/annotations.json",
        classes = ["Kick-off", "Goal"]
    ),
)
log_level = "INFO"  # The level of logging

runner = dict(type="runner_JSON")

visualizer = dict(
    threshold=0.0,
    annotation_range=5000,  # ms
    seconds_to_skip=30,
    scale=1.5,
)
