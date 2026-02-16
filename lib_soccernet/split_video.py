import json
import os
import random
import subprocess

from typing import Optional, Callable
from pathlib import Path

try:
    from utils import prediction_window_mmss, upload_s3_object
except:
    from lib_soccernet.utils import prediction_window_mmss, upload_s3_object


S3_OBJECT_TARGET = "soccernet-v2-amateur"
CLOUDFRONT_DOMAIN = "https://d3tdwb735roscv.cloudfront.net/"


def time_to_seconds(value):
    if isinstance(value, str) and ":" in value:
        parts = [int(p) for p in value.split(":")]
        # MM:SS
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
        # HH:MM:SS
        elif len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        else:
            raise ValueError(f"Invalid time format: {value}")
    return int(float(value))


def split_video_ffmpeg(input_path: str, kick_off: str, output_base: str, suffix: Optional[str] = "") -> None:
    """
    Splits a video into `num_slices` equal parts using FFmpeg.

    Args:
        input_path (str): Path to the input video.
        output_dir (str): Directory to save slices.
        num_slices (int): Number of parts to divide the video into.
    """
    event_sec = time_to_seconds(kick_off)
    start_sec = max(event_sec - 10, 0)
    duration = 5 * 2  # 20 seconds total

    video_path = Path(input_path)

    output_file = os.path.join(
        output_base,
        f"cut_{suffix}_{video_path.stem}{video_path.suffix}"
    )

    # FFmpeg command to cut without re-encoding for speed/quality
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        output_file
    ]

    print(f"Executing: {' '.join(cmd)}")
    print(f"{input_path} clip for {start_sec} saved in {output_file}")
    print(f"Executing.... {subprocess.run(cmd, capture_output=True, text=True)}")


def execute_split_and_save(
        results: dict,
        input_raw_video: str,
        video_id: str,
        data: dict,
        id_file: str,
        model_config: str,
        save_status: Callable
    ) -> None:

    # if for the given inference config this data is not already processed
    # we initialize a fresh list with the clips generated
    if not data[model_config].get("s3_objects_list"):
        data[model_config]["s3_objects_list"] = []

    for result in results:
        for event in result["predictions"]:
            os.makedirs(os.path.join("./", "outputs"), exist_ok=True)
            os.makedirs(os.path.join(f"./outputs/{video_id}"), exist_ok=True)
            timestamp = event["gameTime"]
            label = event["label"]
            confidence = event["confidence"]
            kick_off = prediction_window_mmss(timestamp)
            suffix = f"l:{label}_s:{kick_off}_c:{confidence}"
            split_video_ffmpeg(
                input_raw_video,
                kick_off,
                f"./outputs/{video_id}",
                suffix=suffix
            )

            video_path = Path(input_raw_video)
            file_name = f"cut_{suffix}_{video_path.stem}{video_path.suffix}"
            base_path = f"./outputs/{video_id}"

            output_file = os.path.join(base_path, file_name)

            upload_s3_object(output_file, f"results/{video_id}/{file_name}", S3_OBJECT_TARGET)
            data[model_config]["s3_objects_list"].append(f"{CLOUDFRONT_DOMAIN}results/{video_id}/{file_name}")
            save_status(data, id_file)

    data["status"] = "finished"
    save_status(data, id_file)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a video into N equal parts using FFmpeg.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--kick-off", help="Start of the video to cut")
    parser.add_argument("--end-time", help="End of the video to cut")

    args = parser.parse_args()

    size_video = args.end_time

    try:
        start_time = int(args.kick_off)
    except:
        pass

    split_video_ffmpeg(args.video, args.kick_off, output_base="./")

