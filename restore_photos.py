#!/usr/bin/env python3
"""
Standalone CLI Tool for Photo Restoration & Colorization.
Restores old, blurry, or scratched portrait photos and colorizes vintage black-and-white pictures.

Usage:
  python restore_photos.py -i input.jpg -o restored.jpg
  python restore_photos.py -i input.jpg -o restored.jpg --colorize
  python restore_photos.py -i ./my_old_photos/ -o ./restored_photos/ --colorize
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Ensure virtualenv tools (ffmpeg, etc.) are available on PATH
_script_dir = Path(__file__).resolve().parent
_venv_scripts = _script_dir / ".venv" / "Scripts"
if _venv_scripts.is_dir() and str(_venv_scripts) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_venv_scripts) + os.pathsep + os.environ.get("PATH", "")

from facefusion import core, state_manager
from facefusion.execution import get_available_execution_providers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photo Enhancer & Colorizer CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input image file or directory of images to restore.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output image path or output directory.",
    )
    parser.add_argument(
        "--model",
        default="gfpgan_1.4",
        choices=["gfpgan_1.4", "codeformer", "gfpgan_1.3", "restoreformer_plus_plus"],
        help="Face restoration model.",
    )
    parser.add_argument(
        "--blend",
        type=int,
        default=80,
        help="Enhancement blend percentage (0-100).",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Enable AI photo colorization for black-and-white photos.",
    )
    parser.add_argument(
        "--vibrance",
        action="store_true",
        help="Boost color vibrance and dynamic contrast.",
    )
    parser.add_argument(
        "--beauty",
        action="store_true",
        help="Enable skin beauty smoothing and blemish softening.",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Enable 4x Ultra HD super-resolution upscaling.",
    )
    parser.add_argument(
        "--provider",
        choices=["cuda", "directml", "cpu", "auto"],
        default="auto",
        help="Execution provider for hardware acceleration.",
    )
    return parser.parse_args()


def get_providers(chosen: str) -> List[str]:
    available = get_available_execution_providers()
    if chosen == "auto":
        if "cuda" in available:
            return ["cuda", "cpu"]
        elif "directml" in available:
            return ["directml", "cpu"]
        return ["cpu"]
    return [chosen, "cpu"] if chosen != "cpu" else ["cpu"]


def enhance_single_photo(
    input_path: str,
    output_path: str,
    model: str,
    blend: int,
    colorize: bool,
    vibrance: bool,
    beauty: bool,
    upscale: bool,
    providers: List[str],
) -> bool:
    processors = []
    if colorize:
        processors.append("frame_colorizer")
    processors.append("face_enhancer")
    if upscale:
        processors.append("frame_enhancer")

    # Configure FaceFusion state_manager
    state_manager.init_item("command", "headless-run")
    state_manager.init_item("source_paths", [])
    state_manager.init_item("download_providers", ["github", "huggingface"])
    state_manager.init_item("download_scope", "lite")
    state_manager.init_item("execution_device_ids", [0])
    state_manager.set_item("target_path", input_path)
    state_manager.set_item("output_path", output_path)
    state_manager.set_item("processors", processors)
    state_manager.set_item("face_selector_mode", "many")
    state_manager.set_item("face_enhancer_model", model)
    state_manager.set_item("face_enhancer_blend", blend)
    state_manager.set_item("execution_providers", providers)
    state_manager.set_item("video_memory_strategy", "tolerant")

    if colorize:
        state_manager.set_item("frame_colorizer_model", "ddcolor")
        state_manager.set_item("frame_colorizer_blend", 100)
        state_manager.set_item("frame_colorizer_size", "256x256")
    if upscale:
        state_manager.set_item("frame_enhancer_model", "span_kendata_x4")
        state_manager.set_item("frame_enhancer_blend", 100)

    if not core.common_pre_check() or not core.processors_pre_check():
        print(f"[-] Pre-check failed for {input_path}")
        return False

    code = core.conditional_process()
    success = code == 0 and os.path.isfile(output_path)

    if success and (vibrance or beauty):
        from api_server import enhance_beauty_and_vibrance
        enhance_beauty_and_vibrance(output_path, beauty_retouch=beauty, vibrance_boost=vibrance)

    return success


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    providers = get_providers(args.provider)

    print(f"[*] Hardware Acceleration: {providers[0].upper()}")

    if input_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Processing single photo: {input_path.name}")
        success = enhance_single_photo(
            str(input_path),
            str(output_path),
            args.model,
            args.blend,
            args.colorize,
            args.vibrance,
            args.beauty,
            args.upscale,
            providers,
        )
        if success:
            print(f"[+] Saved restored photo to: {output_path}")
        else:
            print(f"[-] Failed to restore: {input_path}")
            sys.exit(1)

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            print(f"[-] No image files found in {input_path}")
            sys.exit(1)

        print(f"[*] Found {len(files)} photos in directory. Starting batch restoration...")
        succeeded, failed = 0, 0
        for idx, file in enumerate(files, 1):
            out_file = output_path / f"restored_{file.name}"
            print(f"[{idx}/{len(files)}] Restoring {file.name}...")
            if enhance_single_photo(
                str(file),
                str(out_file),
                args.model,
                args.blend,
                args.colorize,
                args.vibrance,
                args.beauty,
                args.upscale,
                providers,
            ):
                succeeded += 1
            else:
                failed += 1

        print(f"[+] Batch complete: {succeeded} succeeded, {failed} failed.")
    else:
        print(f"[-] Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
