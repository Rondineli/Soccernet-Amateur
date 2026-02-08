import os
import boto3

from botocore.exceptions import ClientError


def mmss_to_seconds(time_str: str) -> int:
    """ convert mms string to seconds """
    minutes, seconds = map(int, time_str.split(":"))
    return minutes * 60 + seconds


def seconds_to_mmss(total_seconds: int) -> str:
    """ convert seconds to mms """
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def normalize_id(title: str) -> str:
    """ normalize an id for be able to save with file unix complaint name """
    # Remove spaces, dots, extension and set in lowercase
    return title.replace(" ", "_").replace(".mp4", "").replace(".", "_").lower()


def prediction_window_mmss(prediction_mmss: str, window_s: int = 0) -> str:
    """ compute a given timestamp window (+5s/-5s) """
    pred_sec = mmss_to_seconds(prediction_mmss)

    lower = max(0, pred_sec - window_s)

    return seconds_to_mmss(lower)


def upload_s3_object(file_object: str, file_name: str, s3_bucket: str) -> dict:
    """ Client to upload objects to S3 """
    # Upload the file
    s3_client = boto3.client('s3')

    response = {}

    try:
        response = s3_client.upload_file(
            file_object,
            s3_bucket,
            file_name,
            ExtraArgs={
                "ContentType": "video/mp4"
            }
        )
    except ClientError as e:
        return {"err": str(e)}
    return response
