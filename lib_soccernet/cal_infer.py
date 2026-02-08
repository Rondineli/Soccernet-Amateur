import torch
import os
import logging
import json
import numpy as np

from typing import Optional
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

from mmengine.config import Config, DictAction

from oslactionspotting.apis.inference.builder import build_inferer
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
)
from oslactionspotting.core.utils.io import check_config, whether_infer_split
from oslactionspotting.datasets.builder import build_dataset
from oslactionspotting.models.builder import build_model


WINDOW_PREDIT = 5


def set_logger(work_dir: str) -> logging.Logger:
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)

    log_path = os.path.join(
        work_dir,
        "logs",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    )

    return logging.basicConfig(
        level="DEBUG",
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def extract_time_from_annotation(annotation: str) -> str:
    return annotation.split("-")[-1].strip()


def mmss_to_seconds(time_str: str) -> int:
    minutes, seconds = map(int, time_str.split(":"))
    return minutes * 60 + seconds


def seconds_to_mmss(total_seconds: int) -> str:
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def prediction_window_mmss(prediction_mmss: str, window_s: int = WINDOW_PREDIT):
    pred_sec = mmss_to_seconds(prediction_mmss)

    lower = max(0, pred_sec - window_s)
    # upper = pred_sec + window_s
    return seconds_to_mmss(lower)


def is_prediction_within_window(
    prediction_mmss: str,
    annotation_str: str,
    window_s: int = WINDOW_PREDIT
) -> bool:
    pred_sec = mmss_to_seconds(prediction_mmss)

    annotation_time = extract_time_from_annotation(annotation_str)
    ann_sec = mmss_to_seconds(annotation_time)
    return abs(pred_sec - ann_sec) <= window_s


def transform_annotation(videos: dict) -> dict:
    return  {g["path"]: g["annotations"] for g in videos}


def main(args, cfg):
    logging = set_logger(cfg.work_dir)

    # overwrite cfg from args
    if args.cfg_options is not None:
        print(f"[DEBUG]"" Options: {args.cfg_options}")
        cfg.merge_from_dict(args.cfg_options)

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg.infer_split = whether_infer_split(cfg.dataset.test)


    if not getattr(cfg.model, "load_weights"):
        default_path = os.path.join(cfg.work_dir, "model.pth.tar")
        cfg.model.load_weights = default_path

    # Build Model
    model = build_model(
        cfg,
        default_args=get_default_args_model(cfg),
    )

    default_args = get_default_args_dataset("test", cfg)

    dataset_infer = build_dataset(
        cfg.dataset.test,
        cfg.training.GPU,
        default_args
    )

    inferer = build_inferer(cfg, model)
    inferer.infer(dataset_infer)


def load_results(cfg: object, video_id: Optional[str] = None) -> list:
    if video_id:
        output_challengers = [video_id]
    else:
        output_challengers = ["test", "challenge"]
    base_path = os.path.join(cfg.work_dir, "results_spotting_test")
    _all_results = []

    for bp in output_challengers:
        output_path = os.path.join(base_path, bp)

        for g in os.listdir(output_path):
            _base_path = os.path.join(output_path, g)
            print(f"Checking: {g}")
            if g == "1_ResNet":
                with open(f"{_base_path}/results_spotting.json", "r") as f:
                    _all_results.append(json.loads(f.read()))
                    break

            print(f"Checking second: {_base_path}")
            with open(f"{_base_path}/1_ResNet/results_spotting.json", "r") as f:
                _all_results.append(json.loads(f.read()))

    return _all_results


def load_annotations(cfg) -> json:
    annotation_path = cfg.dataset.test["path"]
    annotations = {}

    with open(annotation_path, "r") as f:
        annotations = json.loads(f.read())

    return transform_annotation(
        annotations["videos"]
    )


def load_video_from_annotations(annotations: dict, video_path: str) -> dict:
    print(annotations)

    try:
        return next(
            annotations[v] for v in annotations.keys()
            if video_path in v
        )
    except StopIteration:
        return []


if __name__ == "__main__":
    parser = ArgumentParser(
        description="context aware loss function",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--weights", type=str, help="Path to specific model weights for inference")
    parser.add_argument("--output", default="./cleaned_output.json", type=str, help="Path to specific model weights for inference")
    parser.add_argument("--video-id", default="", type=str, help="Video Youtube ID")


    # read args
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    check_config(cfg)

    main(args, cfg)
    results = load_results(cfg, args.video_id)
    annotations = load_annotations(cfg)

    _outputs = {}

    print_count = 0
    append_count = 0

    for result in results:
        video_annotations = load_video_from_annotations(annotations, result["Url"])
        for event in result["predictions"]:
            for annotation in video_annotations:
                if is_prediction_within_window(event["gameTime"], annotation["gameTime"]):
                    if event["label"] == annotation["label"]:
                        gae_time_event = event["gameTime"]
                        ant_time_event = annotation["gameTime"]
                        print_count += 1

                        if result["Url"] not in _outputs:
                            _outputs[result["Url"]] = []

                        _outputs[result["Url"]].append(event)
                        
                        append_count += 1
                        break

    print(f"DEBUG: prints={print_count}, appends={append_count}")

    if len(results) == 0 or append_count == 0:
        _outputs = results

    with open(args.output, "w") as f:
        json.dump(_outputs, f)
