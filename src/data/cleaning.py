from typing import Any

import numpy as np
import pandas as pd

def mask_low_conf(channel: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    tmp_channel = channel.copy()
    low = tmp_channel[..., 2] < threshold
    tmp_channel[low, :2] = np.nan
    return tmp_channel

def report_missing(channel: np.ndarray) -> float:
    return float(np.isnan(channel[..., :2]).mean() * 100)

def interpolate(channel: np.ndarray) -> np.ndarray:
    T, C, _ = channel.shape
    flat = channel[..., :2].reshape(T, C * 2)
    filled = (pd.DataFrame(flat).interpolate(limit_direction="both").to_numpy(dtype=float))
    out = channel.copy()
    out[..., :2] = filled.reshape(T, C, 2)
    return out

def clean_channels(channels:dict, threshold: float = 0.1) -> tuple[dict[Any, Any], dict[Any, Any]]:
    cleaned, report = {}, {}
    for channel_name, channel in channels.items():
        masked_channel = mask_low_conf(channel, threshold)
        filled_channel = interpolate(masked_channel)
        cleaned[channel_name] = filled_channel
        report[channel_name] = (report_missing(masked_channel), report_missing(filled_channel))
    return cleaned, report
