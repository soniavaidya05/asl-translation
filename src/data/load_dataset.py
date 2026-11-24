import pandas as pd
import numpy as np
import json
from pathlib import Path

KEYS = {"pose_keypoints_2d":25, "face_keypoints_2d":70, "hand_left_keypoints_2d":21, "hand_right_keypoints_2d":21}

UPPER_BODY_INDEX = [0,1,2,3,4,5,6,7,8,15,16,17,18]

def load_clip(directory: Path) -> dict[str, np.ndarray]:
    out = {k: [] for k in KEYS}
    for frame_path in sorted(directory.glob("*.json")):
        with frame_path.open("r", encoding="utf-8") as f:
            file_json = json.load(f)
            people = file_json["people"]
            person = people[0] if people else None
            for key in KEYS:
                num = KEYS[key]
                if person is not None and len(person.get(key, [])) > 0:
                    arr = np.array(person[key], dtype=np.float32).reshape(num, 3)
                else:
                    arr = np.full((num, 3), np.nan, dtype=np.float32)
                out[key].append(arr)
    return {k: np.stack(v) for k, v in out.items()}

def split_channels(raw_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pose_keypoints_2d = raw_data["pose_keypoints_2d"]
    face_keypoints_2d = raw_data["face_keypoints_2d"]
    hand_left_keypoints_2d = raw_data["hand_left_keypoints_2d"]
    hand_right_keypoints_2d = raw_data["hand_right_keypoints_2d"]


    hands = np.concatenate([hand_left_keypoints_2d,hand_right_keypoints_2d], axis=1)
    body = pose_keypoints_2d[:, UPPER_BODY_INDEX]
    return {"body": body, "hands": hands, "face": face_keypoints_2d}






