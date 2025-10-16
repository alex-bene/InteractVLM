import argparse
import sys
from pathlib import Path

from tinytools import get_logger, setup_prettier_root_logger, setup_prettier_tqdm
from tqdm import tqdm

from model.wrapped_interactvlm import InteractVLM

setup_prettier_root_logger(level="INFO")
setup_prettier_tqdm()
logger = get_logger(__name__)


def parse_args(args: list[str]) -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(description="InteractVLM chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument(
        "--contact_type",
        default="hcontact",
        type=str,
        help="Type of contact prediction: 'hcontact' for 3D human contact, 'h2dcontact' for 2D human contact, 'ocontact'/'oafford' for object contact",  # noqa: E501
    )
    parser.add_argument("--img_folder", default="", type=str)
    parser.add_argument(
        "--input_mode",
        default="folder",
        type=str,
        choices=["folder", "file"],
        help="Input mode: 'folder' for folder-based samples, 'file' for file-based samples (human contact only)",
    )
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


def main(args: list[str]) -> None:  # noqa: D103
    args = parse_args(args)
    img_folder = args.img_folder
    img_folder = Path(img_folder)
    image_paths = []
    object_names = []
    for image_path in list(img_folder.rglob("f0.png")):
        object_name_path = image_path.parent / "object_name.txt"
        if image_path.exists() and object_name_path.exists():
            image_paths.append(image_path)
            object_names.append(object_name_path.read_text().strip())
    logger.info("Found %d images", len(image_paths))
    output_dirs = [image_path.parent for image_path in image_paths]

    pipeline = InteractVLM(
        pretrained_model_name_or_path=args.version,
        vision_tower=args.vision_tower,
        model_max_length=args.model_max_length,
        precision=args.precision,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        image_size=args.image_size,
        local_rank=args.local_rank,
    )

    for image_path, output_dir, object_name in tqdm(
        zip(image_paths, output_dirs, object_names, strict=True), total=len(image_paths), desc="Object fit optimization"
    ):
        try:
            pipeline.forward(
                image_paths=[image_path],
                contact_type=args.contact_type if args.contact_type != "hcontact-ocontact" else "ocontact",
                output_dirs=[output_dir],
                object_names=[object_name],
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                no_progress_bar=True,
            )
            if args.contact_type == "hcontact-ocontact":
                pipeline.forward(
                    image_paths=[image_path],
                    contact_type="hcontact",
                    output_dirs=[output_dir],
                    object_names=[object_name],
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    no_progress_bar=True,
                )
        except Exception:
            logger.exception("Failed to run object fit optimization on %s", image_path)


if __name__ == "__main__":
    main(sys.argv[1:])
