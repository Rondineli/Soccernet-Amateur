import json
import subprocess
import threading
import os
import yt_dlp

from flask import Flask, request, jsonify, redirect
from pathlib import Path
from split_video import execute_split_and_save
from utils import normalize_id

app = Flask(__name__)

DPATH_JSON_DATASET = os.path.join("/datasets/amateur/", "download")
DBASE_PATH_JSON = "/tmp/download_id_status/"
DEFAULT_CONFIG = "configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_st_2.py"
DEFAULT_DATASET_DIR = "/datasets/amateur/"

MODEL_CONFIGS = {
    "json_soccernet_calf_resnetpca512": "configs/contextawarelossfunction/json_soccernet_calf_resnetpca512.py",
    "json_soccernet_calf_resnetpca512_amateur_model_no_tf": "configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_no_tf.py",
    "json_soccernet_calf_resnetpca512_amateur_model_st_2": "configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_st_2.py"
}


os.makedirs(DBASE_PATH_JSON, exist_ok=True)


def download_ph_hook(status: dict) -> None:
    """
    Hook to measure download process status, at every byte downloaded this function is executed
    therefore, persisting download status
    """
    status_download = status.get("status")
    raw_video_id = status.get("info_dict", {}).get("id")
    id_file = raw_video_id
    id_file = os.path.join(DBASE_PATH_JSON, normalize_id(id_file))
    id_file = f"{id_file}.json"

    file_download_name = f"{DEFAULT_DATASET_DIR}download/{raw_video_id}/{raw_video_id}.mp4"

    fragment_index = status.get("fragment_index")
    fragment_count = status.get("fragment_count")

    if fragment_index is not None and fragment_count:
        percent = round((fragment_index / fragment_count) * 100, 2)
    else:
        percent = 0

    db = {
        "filename": id_file,
        "phase_1_status": status_download,
        "phase": 1,
        "file_location": file_download_name,
        "status_completion": percent
    }

    with open(id_file, "w") as f:
        json.dump(db, f)


def start_download(url: str) -> None:
    """
    Step 1 of main handler: Method to start youtubeVideo Id download from youtube API
    """
    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": f"{DEFAULT_DATASET_DIR}download/%(id)s/%(id)s.%(ext)s",
        "progress_hooks": [download_ph_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.add_progress_hook(download_ph_hook)
        ydl.download([url])


def generate_pc512_features(video_download_location: str, data: dict, id_file: str, yt_id: str) -> None:
    """
        Step 2 of main handler: Function to start PCA 512 feature extraction
        * Execute PCA feature extraction of downloaded video
    """
    cmd_feature_extractor = [
        "bash",
        "-x",
        "./execute_features.sh",
        "/OSL-ActionSpotting",
        f"{video_download_location}",
        f"{yt_id}"
    ]
    print(f"Initiated bash command: {' '.join(cmd_feature_extractor) }")
    process = subprocess.Popen(
        cmd_feature_extractor,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/OSL-ActionSpotting",
        env=os.environ.copy(),
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode == 0:
        print(f"IdFile generated, generating features....")
        with open(id_file, "w") as f:
            data["phase"] = 3
            data["pca_file"] = f"{DEFAULT_DATASET_DIR}download/{yt_id}/1_ResNet.npy"
            data["phase_2_status"] = "completed"
            json.dump(data, f)
    else:
        _out = process.stdout.read()
        print(f"[ERROR]: IdFile not generated error: {process.returncode}: {_out}: {process.stderr}")
        raise Exception(f"[ERROR]: IdFile not generated error: {_out}")


def start_inference_results(yt_id: str, annotation_file: str, id_file:str, data: dict, model_config: str) -> None:
    """ 
        Wrapper method of execute_inference to start execution in a threading
        and interpolate data with statuses to save current processing status
    """
    response = execute_inference(yt_id, annotation_file, data, model_config)
    data[model_config] = {}
    data[model_config]["inference_raw_results"] = response
    data["phase"] = 4
    data["phase_3_status"] = "completed"
    with open(id_file, "w") as f:
        json.dump(data, f)


def execute_inference(yt_id: str, annotation_file: str, data: dict, model_config: str):
    """
    Step 5 of main handler:
        * Execute inference of the config parameter (already trained model) collects its result and persist
    """
    result_dir_file = f"{DEFAULT_DATASET_DIR}download/{yt_id}/results.json"

    cmd_feature_extractor = [
        "bash",
        "./execute.sh",
        model_config,
        f"{DEFAULT_DATASET_DIR}download/",
        result_dir_file,
        annotation_file,
        yt_id
    ]
    print(f"Executing: {' '.join(cmd_feature_extractor)}")
    if os.path.exists(annotation_file):
        process = subprocess.Popen(
            cmd_feature_extractor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/OSL-ActionSpotting",
            env=os.environ.copy()
        )
        process.wait()
        if process.returncode == 0:
            with open(result_dir_file, "r") as f:
                data = json.loads(f.read())
                return data
    return {}


def save_status(data: dict, id_file: str) -> None:
    """
        Persist current processing phases and status for a giving video id
        in disk for next consumption
    """
    with open(id_file, "w") as f:
        json.dump(data, f)


def execute_split_and_upload(data: dict, yt_id: str, id_file: str, model_config: str) -> None:
    """
    Step 6 and 7 of the main handler:
        1 - Retrieve json result from inference model
                |
                V
        2 - Iterate of results
                |
                V
        3 - Execute ffmpeg with -5s and +5s (total 10s clip) of current ms of the result label
                |
                V
        4 - Upload generated clip to s3
                |
                V
        5 - Save status
    """
    execute_split_and_save(
        data[model_config]["inference_raw_results"],
        input_raw_video=data["file_location"],
        video_id=yt_id,
        data=data,
        id_file=id_file,
        model_config=model_config,
        save_status=save_status
    )
    data["phase_4_status"] = "completed"
    save_status(data, id_file)


def get_model_config_value(config_rep: str) -> str:
    """ Function to translate friendly config to the path config """
    return MODEL_CONFIGS.get(config_rep, DEFAULT_CONFIG)


@app.route("/status/<file_name>")
def get_download_status(file_name: str) -> jsonify:
    """
    This function receives a file name containing status of a given id exeuction
    loads and return its plain file
    """
    data = '{}'

    id_file = os.path.join(DBASE_PATH_JSON, normalize_id(file_name))
    id_file = f"{id_file}.json"

    data = {
        "status": "not_started_or_not_completed"
    }
    try:
        with open(id_file, "r") as f:
            data = json.loads(f.read())
            return jsonify(data)
    except Exception as e:
        data["msg"] = str(e)
    
    return jsonify(data)


@app.route('/benchmark')
def execute_benchmark() -> jsonify:
    """
    Execute shell script containing mAP@5s and mAP@10s for each of model config
    trained and saved for this study
    """
    return Exception("NotImplementedErrorYet")


@app.route("/<yt_id>/")
def download_youtube(yt_id):
    """ 
    Method route to download youtube video using yt-dl

    This method is designed following similar state machine design, where each action contain an state to 
    allow next status to be execute, in this function case:

        1 - Download video if not downloaded
                |
                V
        2 - If video downloaded run PCA 512-dimension feature extraction
                |
                V
        3 - If the config parameters alread processed - return values
                |
                V
        4 - If not, create annotation for inference execution
                |
                V
        5 - Start Inference execution and retrieve status
                |
                V
        6 - Generate 10s clips using ffmepg for each annotated result
                |
                V
        7 - Upload to s3
                |
                V
        8 - Return
    """
    # converting youtubeId to the status file saved in disk
    id_file = os.path.join(DBASE_PATH_JSON, normalize_id(yt_id))
    id_file = f"{id_file}.json"

    model_config_arg = request.args.get("modelConfig", "")
    model_config = get_model_config_value(model_config_arg)

    model_name = Path(model_config).stem

    # Checking if we already downloaded this video and or 
    # already exists a json file with processing status
    if os.path.exists(id_file):
        with open(id_file, "r") as f:
            data = json.load(f)
        
        if model_config not in data.get("model_name", []):
            # if this model config inference is not already present
            # we should restart the phases controlers to process a new 
            # inference with a new model for an already downloaded video and 
            # pca extracted, meaning, we should re-start from step 3 forward
            try:
                # if this key does not exists, it is fine, we should move on
                # it should delete only if:
                #   * the new inference is not in the json file
                #   * and a prior inference already ran status
                del data["phase_3_status"]
                del data["phase_4_status"]
            except:
                pass

            # Save the models names for future consumption
            try:
                data["model_name"].append(model_config)
            except:
                data["model_name"] = [model_config]

        if data.get("phase_1_status") == "downloading":
            # if we have an already phase 1 (video download)
            # in progress, redirect request to consult json status
            return redirect(f"/status/{yt_id}")

        # check status once it is finished
        if data.get("percentage") == 100 or data.get("phase_1_status") == "finished":
            os.makedirs(f"{DEFAULT_DATASET_DIR}download/{yt_id}", exist_ok=True)
            data["phase"] = 2
            # if download completed - save status for the phase 2 (PCA)
            save_status(data, id_file)

            video_download_location = data["file_location"]

            if not data.get("pca_file") and not data.get("phase_2_status"):
                # the PCA can take sometimes more than 20m, so it should be in background to not cause timeouts
                # once it is started the handler will redirect it to the status endpoint
                # so the consumer can follow up once it is finished to convert
                t = threading.Thread(
                    target=generate_pc512_features,
                    args=(video_download_location, data, id_file, yt_id),
                    daemon=True  # dies when app exits
                )
                t.start()
                # persist new phase progress
                data["phase_2_status"] = "started"
                save_status(data, id_file)

                # finally, if we already started the threading for pca feature extraction,
                # return the redirect page
                return redirect(f"/status/{yt_id}")

            # Those elifs will check if there is a alreayd started process
            # if so, we should redirect it to the status page
            elif data.get("phase_2_status") == "started":
                return redirect(f"/status/{yt_id}")

            elif data.get("phase_3_status") == "started":
                return redirect(f"/status/{yt_id}")
            
            elif data.get("phase_4_status") == "started":
                return redirect(f"/status/{yt_id}")

            else:
                # if not redirect up to here, we have a mp4 and a .npy file, ready to run the inference
                # we will save an temporarilly annotation.json file containing the file location
                # to the inference grab the location of the .npy file
                print(f"PCA Completed....returning now: {data.get('pca_file')}")
                annotation_file = f"{DEFAULT_DATASET_DIR}download/{yt_id}/{yt_id}_annotation.json"
                if not os.path.exists(annotation_file):
                    with open(annotation_file, "w") as f:
                        annotation_data = {
                            "version": 1,
                            "date": "2025.29.12",
                            "videos": [
                                {
                                    "path": f"{yt_id}/1_ResNet.npy",
                                    "annotations": [],
                                    "input_type": "features"
                                }
                            ],
                            "labels": [
                                "Kick-off",
                                "Goal"
                            ]
                        }
                        json.dump(annotation_data, f)

                    # once annotation is saved, we are ready to start next phase
                    # running the inference
                    data["phase"] = 3
                    data["annotation_location"] = annotation_file
                    save_status(data, id_file)

                if not data.get("phase_3_status") and data.get("phase_2_status") == "completed":
                    # as inference can take more than a minute to run to avoid timeouts
                    # we are starting the process in background then return a redirect
                    # to the consumer be able query the current status
                    data["phase_3_status"] = "started"
                    save_status(data, id_file)

                    t = threading.Thread(
                        target=start_inference_results,
                        args=(yt_id, annotation_file, id_file, data, model_config),
                        daemon=True  # dies when app exits
                    )
                    t.start()
                
                if data.get("phase_3_status") == "completed" and not data.get("phase_4_status"):
                    # once step 3 has finished, we will execute ffmpeg to retrieve 10s video
                    # of each results (-5s < current timestamp > +5s - total 10s)
                    # and upload it to an s3 to persist inference results
                    data["phase_4_status"] = "started"
                    save_status(data, id_file)

                    t = threading.Thread(
                        target=execute_split_and_upload,
                        args=(data, yt_id, id_file, model_config),
                        daemon=True  # dies when app exits
                    )
                    t.start()

                if data.get("phase_4_status") == "completed":
                    # if ecverything is completed, just return its json execution
                    # nothing else to do
                    return jsonify(data)
                else:
                    # while it is processing, return redirect endpoint for 
                    # querying status
                    return redirect(f"/status/{yt_id}")

    # if there is no file already saved, it means
    # we should start a download process and save its status
    # so next request, will contain a file with next steps
    url = f"https://www.youtube.com/watch?v={yt_id}"

    t = threading.Thread(
        target=start_download,
        args=(url,),
        daemon=True  # dies when app exits
    )
    t.start()

    return redirect(f"/status/{yt_id}")


if __name__ == "__main__":
    app.run(debug=True, port=5002)
