import torch

def event_segmentation(video_feats, model):
    """
    Simulates event-level segmentation using a boundary predictor model.

    Args:
        video_feats (Tensor): shape (T, D)
        model: object with predict_boundaries(video_feats) -> list of ints

    Returns:
        list[Tensor]: segmented chunks of video_feats
    """
    boundaries = model.predict_boundaries(video_feats)  # e.g., [0, 25, 48, 80]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segments.append(video_feats[start:end])
    return segments
