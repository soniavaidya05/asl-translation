from typing import Any

import pandas as pd
from pathlib import Path

def load_translations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = df.columns.str.strip()
    df["SENTENCE"] = df["SENTENCE"].str.strip()
    return df

def sentence_lookup(df: pd.DataFrame) -> dict[Any, Any]:
    return dict(zip(df["SENTENCE_NAME"], df["SENTENCE"]))

def matching(keypoints_directory: Path, lookup) -> tuple[dict[Any, Any], set[Any], set[Any]]:
    files = {p.name for p in keypoints_directory.iterdir() if p.is_dir()}
    found_files = {}
    not_found_files = set()
    csv_only_files = set()
    for file in files:
        if file in lookup:
            found_files[file] = {
                "clip_dir": keypoints_directory / file,
                "sentence": lookup[file]
            }
        else:
            not_found_files.add(file)
    for key in lookup:
        if key not in files:
            csv_only_files.add(key)
    return found_files, not_found_files, csv_only_files