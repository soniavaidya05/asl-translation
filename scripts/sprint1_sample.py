import sys
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.matching import load_translations, sentence_lookup, matching
from src.data.load_dataset import load_clip, split_channels
from src.data.cleaning import clean_channels
from src.data.normalize import normalize_body, normalize_hands, normalize_face, zero_fill
from src.models.embed import ChannelEmbedding, MultiStreamEmbedding
import torch


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "preprocess.yaml"))
    keypoints_dir = ROOT / cfg["paths"]["keypoints_dir"]
    csv_path = ROOT / cfg["paths"]["translations_csv"]
    clip_id = cfg["sample_clip"]

    df = load_translations(csv_path)
    lookup = sentence_lookup(df)
    found, folder_only, csv_only = matching(keypoints_dir, lookup)
    print(f"matched: {len(found)}  folder-only: {len(folder_only)}  csv-only: {len(csv_only)}")

    info = found[clip_id]
    print(f"\nclip: {clip_id}")
    print(f"sentence: {info['sentence']}")

    raw = load_clip(info["clip_dir"])
    chans = split_channels(raw)
    for name, arr in chans.items():
        print(f"{name:6s} {arr.shape}")

    cleaned, report = clean_channels(chans, cfg["cleaning"]["conf_threshold"])
    for name, (before, after) in report.items():
        print(f"{name:6s} missing {before:5.1f}% -> {after:5.1f}%  shape {cleaned[name].shape}")

    normed = {
        "body": normalize_body(cleaned["body"]),
        "hands": normalize_hands(cleaned["hands"]),
        "face": normalize_face(cleaned["face"]),
    }
    normed = zero_fill(normed)
    for k, v in normed.items():
        print(
            f"{k:6s} {v.shape}  min {np.nanmin(v):6.2f}  max {np.nanmax(v):6.2f}  NaN {np.isnan(v).mean() * 100:.1f}%")

    inputs = {k: torch.from_numpy(v).float().reshape(1, v.shape[0], -1) for k, v in normed.items()}
    in_dims = {k: v.shape[-1] for k, v in inputs.items()}
    model = MultiStreamEmbedding(in_dims)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    for k in {"body", "hands", "face"}:
        print(f"{k:6s}  input {tuple(inputs[k].shape)} " f"embedded {tuple(outputs[k].shape)}")

if __name__ == "__main__":
    main()