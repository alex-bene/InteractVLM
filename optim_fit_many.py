"""Single frame pipeline for object localization."""

import argparse
import random
import subprocess
import sys
from pathlib import Path

from rich.traceback import install
from tinytools import setup_prettier_root_logger, setup_prettier_tqdm
from tinytools.logger import get_logger
from tqdm import tqdm

install(show_locals=False)
random.seed(42)

logger = get_logger(__name__)

setup_prettier_tqdm()
setup_prettier_root_logger("info", rich_handler_kwargs={"tracebacks_show_locals": False})

def parse_args(args: list[str]) -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(description="InteractVLM Optimization")
    parser.add_argument("--results_dir", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # 0. Config
    results_dir = args.results_dir
    skip_existing_results = True
    max_videos = None

    # 1. Load data
    results_dir = Path(results_dir)
    videos_to_optim = [
        item.parent.parent.name
        for item in results_dir.rglob("f0_hcontact_vertices.npz")
        if item.with_name("f0_oafford_vertices.npz").exists()
        and (not skip_existing_results or not (item.parent / "optim_fit_results" / "final.obj").exists())
    ]
    logger.info("Found %d videos to run optim.fit", len(videos_to_optim))
    videos_to_optim = sorted(videos_to_optim)[:max_videos]
    logger.info("Using %d videos to run optim.fit", len(videos_to_optim))
    logger.info("First 5 videos: %s", videos_to_optim[:5])

    images_paths = [results_dir / v_name / "f0" / "f0.png" for v_name in videos_to_optim]

    # 4. Estimate object mesh scaling
    logger.info("Running object fit optimization 🚧 ...")
    successes = 0
    for image_path in tqdm(images_paths, desc="Object fit optimization"):
        try:
            # run python script in ../optim/fit.py with args --input_path image_path --cfg optim/cfg/fit.yaml
            subprocess.run(
                ["python", "-m", "optim.fit", "--input_path", str(image_path), "--cfg", "optim/cfg/fit.yaml"],
                check=True,
            )
            successes += 1
        except Exception:
            logger.exception("Failed to run object fit optimization on %s", image_path)

    logger.info("Finished object fit optimization. %d successes out of %d", successes, len(images_paths))
