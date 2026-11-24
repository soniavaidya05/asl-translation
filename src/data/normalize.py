import numpy as np

NECK, R_SHOULDER, L_SHOULDER = 1, 2, 5
WRIST, MIDDLE_MCP = 0, 9
FACE_NOSE_TIP, FACE_EYE_R, FACE_EYE_L = 30, 36, 45

def _dist(positions, a, b):
    return np.linalg.norm(positions[:, a] - positions[:, b], axis=1)

def _center_scale(positions, center, scale):
    scale = np.where((scale < 1e-6) | np.isnan(scale), np.nanmedian(scale), scale)
    positions = positions - positions[:, center:center + 1]
    return positions/scale[:, None, None]

def normalize_body(body):
    positions = body[..., :2]
    scale = _dist(positions, R_SHOULDER, L_SHOULDER)
    return _center_scale(positions, center=NECK, scale=scale)

def normalize_face(face):
    positions = face[..., :2]
    scale = _dist(positions, FACE_EYE_R, FACE_EYE_L)
    return _center_scale(positions, center=FACE_NOSE_TIP, scale=scale)

def normalize_hands(hand):
    positions = hand[..., :2]
    left, right = positions[:, :21], positions[:, 21:]

    left_scale = _dist(left, WRIST, MIDDLE_MCP)
    right_scale = _dist(right, WRIST, MIDDLE_MCP)

    left_n = _center_scale(left, center=WRIST, scale=left_scale)
    right_n = _center_scale(right, center=WRIST, scale=right_scale)

    return np.concatenate((left_n, right_n), axis=1)

def zero_fill(chans: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.nan_to_num(v, nan=0.0) for k, v in chans.items()}