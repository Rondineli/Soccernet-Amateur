import json
import os
import glob
import cv2
from tqdm import tqdm
import argparse

# ================= CONFIG =================
VIDEO_PATH_ROOT = "/opt/projects/datasets/"
OUTPUT_DIR = "context_frames"
CONTEXT_OFFSETS_SEC = [-15, 0, 15]  # 5 screenshots
# ==========================================


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def strip_model_from_url(url):
    return os.path.dirname(url)


def find_video(video_dir):
    mp4s = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No MP4 found in {video_dir}")
    if len(mp4s) > 1:
        print(f"[WARN] Multiple MP4s found, using first: {mp4s[0]}")
    return mp4s[0]




def extract_frames(video_path, event_ms, output_dir, label, confidence):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    event_sec = event_ms / 1000.0

    for offset in CONTEXT_OFFSETS_SEC:
        t = event_sec + offset
        if t < 0:
            t = 4

        frame_idx = int(t * fps)
        if frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue
        fname = (
            f"{label}_t{event_ms}ms_offset{offset:+d}s_"
            f"conf{confidence:.3f}.jpg"
        )
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, frame)
    
    cap.release()

def main(args):
    data = load_json(args.result_json_file)

    url = data["Url"]
    predictions = data["predictions"]

    print(f"Loading file: {url} with {len(predictions)} predictions")

    base_video_dir = strip_model_from_url(url)
    full_video_dir = os.path.join(VIDEO_PATH_ROOT, base_video_dir)

    print(f"[INFO] Video directory: {full_video_dir}")
    video_path = find_video(full_video_dir)
    print(f"[INFO] Using video: {video_path}")

    output_dir_test = os.path.join(
        strip_model_from_url(args.result_json_file),
        OUTPUT_DIR
    )
    print(f"output will be written in {output_dir_test}")
    #return

    os.makedirs(output_dir_test, exist_ok=True)

    for i, pred in enumerate(tqdm(predictions, desc="Processing predictions")):
        label = pred["label"]
        position_ms = pred["position"]
        confidence = pred.get("confidence", 0.0)

        if confidence < 0.2 and label.lower() == "goal":
            continue

        event_dir = os.path.join(
            output_dir_test, f"{label}_{position_ms}ms"
        )
        os.makedirs(event_dir, exist_ok=True)
        print(f"Saving new frame at {event_dir}")

        extract_frames(
            video_path,
            position_ms,
            event_dir,
            label,
            confidence
        )
        if i == 2:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json-file", required=True, help="Path to SoccerNet json result to extract frames")
    args = parser.parse_args()

    main(args)
