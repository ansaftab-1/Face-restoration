"""
18+ / NSFW Content Restriction & Moderation Library for FaceFusion
Uses the 3-model ONNX ensemble from `facefusion-assets` (`nsfw_1.onnx` EraX, `nsfw_2.onnx` Marqo, `nsfw_3.onnx` Freepik)
to block 18+ / adult photos and videos at upload time and run time.
"""

import os
from typing import Any, Dict, List, Optional

import numpy

from facefusion import content_analyser, logger
from facefusion.filesystem import filter_image_paths, is_image, is_video
from facefusion.vision import detect_video_fps, is_vision_frame, read_image, restrict_trim_frame

REJECTION_MESSAGE = (
    "[18+ Content Restricted] Adult / NSFW (18+) content is strictly prohibited. "
    "Your photo/video has been rejected and cannot be uploaded or processed. "
    "(18+ content upload ya run karna mana hai - photo reject kar di gayi hai.)"
)


def inspect_frame_scores(vision_frame: Any) -> Dict[str, Any]:
    """
    Runs the 3 NSFW ONNX models (`nsfw_1`, `nsfw_2`, `nsfw_3`) on a vision frame
    and returns individual flags, raw scores, and the final 18+ verdict.
    """
    content_analyser.pre_check()

    # Model 1: EraX NSFW Detector (threshold > 0.2)
    frame_1 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_1")
    det_1 = content_analyser.forward_nsfw(frame_1, "nsfw_1")
    score_1 = float(numpy.max(numpy.amax(det_1[:, 4:], axis=1)))
    flag_1 = bool(score_1 > 0.2)

    # Model 2: Marqo NSFW Classifier (threshold > 0.25)
    frame_2 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_2")
    det_2 = content_analyser.forward_nsfw(frame_2, "nsfw_2")
    score_2 = float(det_2[0] - det_2[1])
    flag_2 = bool(score_2 > 0.25)

    # Model 3: Freepik NSFW Classifier (threshold > 10.5)
    frame_3 = content_analyser.prepare_detect_frame(vision_frame, "nsfw_3")
    det_3 = content_analyser.forward_nsfw(frame_3, "nsfw_3")
    score_3 = float((det_3[2] + det_3[3]) - (det_3[0] + det_3[1]))
    flag_3 = bool(score_3 > 10.5)

    is_18_plus = bool((flag_1 and flag_2) or (flag_1 and flag_3) or (flag_2 and flag_3))

    return {
        "is_18_plus": is_18_plus,
        "allowed": not is_18_plus,
        "models": {
            "nsfw_1_erax": {"flagged": flag_1, "score": round(score_1, 4), "threshold": 0.2},
            "nsfw_2_marqo": {"flagged": flag_2, "score": round(score_2, 4), "threshold": 0.25},
            "nsfw_3_freepik": {"flagged": flag_3, "score": round(score_3, 4), "threshold": 10.5},
        },
        "message": REJECTION_MESSAGE if is_18_plus else "Safe content (SFW) - allowed.",
    }


def check_media_18_plus(
    file_path: str,
    trim_frame_start: Optional[int] = None,
    trim_frame_end: Optional[int] = None,
    delete_on_reject: bool = False,
) -> Dict[str, Any]:
    """
    Checks an uploaded image or video file for 18+ / NSFW content.
    If 18+ content is detected and `delete_on_reject=True`, deletes the file immediately.
    """
    if not file_path or not os.path.isfile(file_path):
        return {
            "is_18_plus": False,
            "allowed": True,
            "media_type": "unknown",
            "message": "File not found or non-visual file.",
        }

    content_analyser.pre_check()

    if is_image(file_path):
        vision_frame = read_image(file_path)
        if not is_vision_frame(vision_frame):
            return {
                "is_18_plus": False,
                "allowed": True,
                "media_type": "image",
                "message": "Unable to decode image frame.",
            }
        details = inspect_frame_scores(vision_frame)
        details["media_type"] = "image"
        if details["is_18_plus"]:
            logger.error(REJECTION_MESSAGE, __name__)
            if delete_on_reject:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return details

    if is_video(file_path):
        start_f, end_f = restrict_trim_frame(file_path, trim_frame_start, trim_frame_end)
        flagged = content_analyser.analyse_video(file_path, start_f, end_f)
        if flagged:
            logger.error(REJECTION_MESSAGE, __name__)
            if delete_on_reject:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return {
            "is_18_plus": bool(flagged),
            "allowed": not bool(flagged),
            "media_type": "video",
            "message": REJECTION_MESSAGE if flagged else "Safe video content (SFW) - allowed.",
        }

    return {
        "is_18_plus": False,
        "allowed": True,
        "media_type": "other",
        "message": "Safe content - allowed.",
    }


def check_source_and_target(
    source_paths: Optional[List[str]],
    target_path: Optional[str],
    trim_frame_start: Optional[int] = None,
    trim_frame_end: Optional[int] = None,
    delete_on_reject: bool = False,
) -> Dict[str, Any]:
    """
    Validates both source image(s) and target image/video for 18+ content.
    Returns immediately on the first rejected file.
    """
    for src_path in filter_image_paths(source_paths or []):
        res = check_media_18_plus(src_path, delete_on_reject=delete_on_reject)
        if res["is_18_plus"]:
            if delete_on_reject and target_path and os.path.isfile(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            res["rejected_input"] = "source"
            res["rejected_file"] = os.path.basename(src_path)
            return res

    if target_path:
        res = check_media_18_plus(
            target_path,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            delete_on_reject=delete_on_reject,
        )
        if res["is_18_plus"]:
            if delete_on_reject:
                for src_path in source_paths or []:
                    if src_path and os.path.isfile(src_path):
                        try:
                            os.remove(src_path)
                        except OSError:
                            pass
            res["rejected_input"] = "target"
            res["rejected_file"] = os.path.basename(target_path)
            return res

    return {
        "is_18_plus": False,
        "allowed": True,
        "rejected_input": None,
        "message": "Both source and target passed 18+ content moderation.",
    }
