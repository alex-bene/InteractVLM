import logging
import shutil
from pathlib import Path
from typing import Literal, cast

import cv2
import joblib as jl
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.rich import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from datasets.base_contact_dataset import normalize_cam_params
from model.InteractVLM import InteractVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from preprocess_data.constants import HUMAN_VIEW_DICT, OBJS_VIEW_DICT, SMPL_TO_SMPLX_MAPPING
from utils.demo_utils import generate_sam_inp_objs, process_object_mesh_with_contacts, process_smplx_mesh_with_contacts
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    convert_contacts,
)

logger = logging.getLogger(__name__)


def preprocess(
    x: torch.Tensor, pixel_mean: torch.Tensor | None = None, pixel_std: torch.Tensor | None = None, img_size: int = 1024
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    pixel_mean = pixel_mean or torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = pixel_std or torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    h, w = x.shape[-2:]
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad and return
    return F.pad(x, (0, img_size - w, 0, img_size - h))


class InteractVLM:  # noqa: D101
    def __init__(
        self,
        pretrained_model_name_or_path: str = "xinlai/LISA-13B-llama2-v1",
        vision_tower: str = "openai/clip-vit-large-patch14",
        model_max_length: int = 512,
        precision: Literal["fp32", "bf16", "fp16"] = "bf16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        image_size: int = 1024,
        local_rank: int = 0,
    ) -> None:
        self.precision = precision
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.setup_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            vision_tower=vision_tower,
            model_max_length=model_max_length,
            precision=precision,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            image_size=image_size,
            local_rank=local_rank,
        )

    def setup_model(  # noqa: D102
        self,
        pretrained_model_name_or_path: str,
        vision_tower: str,
        model_max_length: int,
        precision: Literal["fp32", "bf16", "fp16"],
        load_in_4bit: bool,
        load_in_8bit: bool,
        image_size: int,
        local_rank: int,
    ) -> None:
        # Create model
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.smpl_to_smlpx_mapping = jl.load(SMPL_TO_SMPLX_MAPPING)["matrix"]
        self.smpl_to_smlpx_mapping = torch.tensor(self.smpl_to_smlpx_mapping).float().cuda()

        torch_dtype = torch.float32
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"], load_in_8bit=True
                    ),
                }
            )

        kwargs.update({"train_from_LISA": False})
        kwargs.update({"train_from_LLAVA": False})

        self.model = InteractVLMForCausalLM.from_pretrained(
            pretrained_model_name_or_path, low_cpu_mem_usage=True, vision_tower=vision_tower, **kwargs
        )

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        self.vision_tower = self.model.get_model().get_vision_tower()
        self.vision_tower.to(dtype=torch_dtype)

        if precision == "bf16":
            self.model = self.model.bfloat16()
        elif precision == "fp16" and (not load_in_4bit) and (not load_in_8bit):
            self.vision_tower = self.model.get_model().get_vision_tower()
            self.model.model.vision_tower = None
            import deepspeed

            self.model_engine = deepspeed.init_inference(
                model=self.model, dtype=torch.half, replace_with_kernel_inject=True, replace_method="auto"
            )
            self.model = self.model_engine.module
            self.model.model.vision_tower = vision_tower.half().cuda()
        elif precision == "fp32":
            self.model = self.model.float()

        if not load_in_8bit and not load_in_4bit and precision != "fp16":
            self.model = self.model.cuda()

        self.vision_tower = self.model.get_model().get_vision_tower()
        self.vision_tower.to(device=local_rank)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(image_size)

        self.model.eval()

    def cast_to_precision(self, *args: torch.Tensor, precision: str | None = None) -> list[torch.Tensor] | torch.Tensor:
        """Cast the given arguments to the specified precision.

        Assumes all arguments are torch.Tensors.
        """
        precision = precision or self.precision
        if precision == "bf16":
            return [arg.bfloat16() for arg in args] if len(args) > 1 else args[0].bfloat16()
        if precision == "fp16":
            return [arg.half() for arg in args] if len(args) > 1 else args[0].half()
        return [arg.float() for arg in args] if len(args) > 1 else args[0].float()

    def prepare_camera_params(self, camera_params: dict) -> tuple[torch.Tensor, list[str]]:  # noqa:D102
        view_names = list(camera_params.keys())
        cam_params = [normalize_cam_params(camera_params[view]) for view in view_names]
        return torch.stack(cam_params).unsqueeze(0).cuda(), view_names

    def prepare_inputs(  # noqa: D102
        self,
        contact_type: Literal["hcontact", "hcontact-wScene", "oafford", "hcontact-ocontact", "h2dcontact"],
        image_paths: list[Path],
        object_names: list[str],
    ) -> dict[str, list[str], list[Path], torch.Tensor]:
        ######################################### Prediciting Object Contact #########################################

        if "oafford" in contact_type or "ocontact" in contact_type:
            BASE_PROMPT = '"What type of affordance does the human-object interaction suggest? Then, segment the area on the {class_name} where the human is making contact.",'  # noqa: E501
            ## Get camera params
            cam_params, view_names = self.prepare_camera_params(
                OBJS_VIEW_DICT[self.model.config.oC_sam_view_type]["cam_params"]
            )
            ## Prepare prompts
            prompts = []
            sam_image_paths = []
            overlay_sam_paths = []
            lift2d_dict_paths = []
            for image_path, object_name in zip(image_paths, object_names, strict=True):
                prompts.append(BASE_PROMPT.format(class_name=object_name))
                sam_base_folder = image_path.parent / "sam_inp_objs"

                ### Check if sam_inp_objs folder exists, if not generate it
                if not sam_base_folder.exists():
                    obj_mesh_path = image_path.parent / "object_mesh.obj"
                    if obj_mesh_path.exists():
                        logger.debug("sam_inp_objs not found, generating for %s", image_path.parent)
                        generate_sam_inp_objs(obj_mesh_path)
                    else:
                        logger.warning("object_mesh.obj not found at %s, cannot generate sam_inp_objs", obj_mesh_path)
                        continue

                sam_image_paths.append([sam_base_folder / f"obj_render_color_{view}.png" for view in view_names])
                overlay_sam_paths.append([sam_base_folder / f"obj_render_grey_{view}.png" for view in view_names])
                lift2d_dict_paths.append(sam_base_folder / "lift2d_dict.pkl")
            # Size check
            if not (
                len(prompts)
                == len(image_paths)
                == len(sam_image_paths)
                == len(overlay_sam_paths)
                == len(lift2d_dict_paths)
            ):
                msg = "Number of prompts, llava images and sam images must be same"
                raise ValueError(msg)

        ######################################### Prediciting 2D Human Contact #########################################
        elif "h2dcontact" in contact_type:
            BASE_PROMPT = (
                "Segment the area on the human's body that is in direct contact with the {object} in this image."
            )
            # Get camera params
            cam_params = torch.rand(1, 5).cuda()  # Dummy camera parameters for 2D
            # Prepare prompts
            prompts = []
            for object_name in object_names:
                prompts.append(BASE_PROMPT.format(object=object_name))
            # Prepare other paths
            sam_image_paths = [None] * len(image_paths)
            overlay_sam_paths = [None] * len(image_paths)
            lift2d_dict_paths = [None] * len(image_paths)
            # Size check
            if len(prompts) != len(image_paths):
                msg = "Number of prompts and llava images must be same"
                raise ValueError(msg)

        ######################################### Prediciting Human Contact #########################################
        elif "hcontact" in contact_type:
            BASE_PROMPT = "Which body parts are in contact with the {object}? Segment these contact areas."
            # Get camera params
            cam_params, view_names = self.prepare_camera_params(
                HUMAN_VIEW_DICT[self.model.config.hC_sam_view_type]["cam_params"]
            )
            # Prepare prompts
            prompts = []
            for object_name in object_names:
                prompts.append(BASE_PROMPT.format(object=object_name))
            # Prepare other paths
            base_path = Path("./data/hcontact_vitruvian/")
            sam_image_paths = [[base_path / f"body_render_norm_{view}.png" for view in view_names]] * len(image_paths)
            overlay_sam_paths = [[base_path / f"smplh_body_render_blue_{view}.png" for view in view_names]] * len(
                image_paths
            )
            lift2d_dict_paths = [None] * len(image_paths)
            # Size check
            if not (len(prompts) == len(image_paths) == len(sam_image_paths)):
                msg = "Number of prompts, llava images and sam images must be same"
                raise ValueError(msg)

        return {
            "prompts": prompts,
            "sam_image_paths": sam_image_paths,
            "overlay_sam_paths": overlay_sam_paths,
            "cam_params": cam_params,
            "lift2d_dict_paths": lift2d_dict_paths,
        }

    def forward(  # noqa: D102
        self,
        image_paths: list[str | Path],
        contact_type: Literal["hcontact", "hcontact-wScene", "oafford", "hcontact-ocontact", "h2dcontact"],
        output_dirs: list[str | Path],
        object_names: list[str],
        conv_type: Literal["llava_v1", "llava_llama_2"] = "llava_v1",
        use_mm_start_end: bool = True,
        no_progress_bar: bool = False,
    ) -> None:
        image_paths = [Path(image_path) for image_path in image_paths]
        output_dirs = [Path(output_dir) for output_dir in output_dirs]

        prompts, sam_image_paths, overlay_sam_paths, cam_params, lift2d_dict_paths = self.prepare_inputs(
            contact_type, image_paths, object_names
        ).values()

        pbar = tqdm(
            total=len(image_paths),
            desc=f"InteractVLM {contact_type}",
            disable=no_progress_bar,
            unit="image",
            unit_scale=True,
        )
        for prompt_in, image_path, sam_image_path, overlay_sam_path, lift2d_dict_path, output_dir in zip(
            prompts, image_paths, sam_image_paths, overlay_sam_paths, lift2d_dict_paths, output_dirs, strict=True
        ):
            # Check if file exists
            image_path = cast("Path", image_path)
            if not image_path.exists():
                logger.warning("File not found in %s", image_path)
                pbar.update(1)
                continue
            # Create output folder
            output_dir = cast("Path", output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Prepare conversation
            conv = conversation_lib.conv_templates[conv_type].copy()
            conv.messages = []
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_in
            if use_mm_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            # Load image
            image_llava_np = cv2.imread(image_path)
            image_llava_np: np.ndarray = cv2.cvtColor(image_llava_np, cv2.COLOR_BGR2RGB)
            image_clip: torch.Tensor = (
                self.clip_image_processor.preprocess(image_llava_np, return_tensors="pt")["pixel_values"][0]
                .unsqueeze(0)
                .cuda()
            )
            image_clip = self.cast_to_precision(image_clip)

            # Handle 2D contact case differently
            if "h2dcontact" in contact_type:
                # For 2D contact, use the original image as SAM input
                orig_size_list = [image_llava_np.shape[:2]]
                sam_img = self.transform.apply_image(image_llava_np)
                resize_list = [sam_img.shape[:2]]
                sam_img = torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()
                sam_multiview = preprocess(sam_img).unsqueeze(0).unsqueeze(0).cuda()
            else:
                # Original 3D contact processing
                sam_image = [Image.open(sam_img) for sam_img in sam_image_path]
                sam_multiview = np.stack([np.asarray(sam_img) for sam_img in sam_image], axis=0)
                valid_masks_region = [(sam_img.sum(axis=-1) < 255 * 3).astype(np.uint8) for sam_img in sam_multiview]
                sam_multiview = [self.transform.apply_image(sam_img) for sam_img in sam_multiview]
                resize_list = [sam_multiview[0].shape[:2]]
                sam_multiview = torch.stack(
                    [preprocess(torch.from_numpy(sam_img).permute(2, 0, 1).contiguous()) for sam_img in sam_multiview]
                )
                sam_multiview = sam_multiview.unsqueeze(0).cuda()

            # Prepare model inputs
            sam_multiview, cam_params = self.cast_to_precision(sam_multiview, cam_params)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()
            # Run model inference
            output = self.model.evaluate(
                image_clip,
                sam_multiview,
                input_ids,
                cam_params,
                resize_list,
                original_size_list=resize_list if "h2dcontact" not in contact_type else orig_size_list,
                lift2d_dict_path=lift2d_dict_path,
                contact_type=contact_type,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )
            output_ids, pred_masks = output["output_ids"], output["pred_masks"]
            if "h2dcontact" in contact_type:
                self.post_process_h2dcontact(output_ids, pred_masks, image_path, output_dir)
            else:
                self.post_process(
                    output_ids,
                    pred_masks,
                    output["pred_contact_3d"],
                    contact_type,
                    image_path,
                    output_dir,
                    valid_masks_region,
                    overlay_sam_path,
                )
            pbar.update(1)
        pbar.close()

    def post_process_h2dcontact(  # noqa: D102
        self,
        output_ids: torch.Tensor,
        pred_masks: torch.Tensor,
        image_path: Path,
        output_dir: Path,
        image_llava_np: np.ndarray,
    ) -> None:
        # Decode the output text
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        pred_masks = pred_masks[0][0]

        if pred_masks is not None:
            binary_mask = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)

            alpha = 0.6
            mask_color_cyan = (np.array([0.0, 1.0, 1.0]) * 255).astype(np.uint8)  # Cyan
            mask_color_red = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)

            # Save original image
            cv2.imwrite(output_dir / "image.png", cv2.cvtColor(image_llava_np, cv2.COLOR_RGB2BGR))

            # Red overlay
            overlay_image_red = image_llava_np.copy()
            overlay_image_red[binary_mask == 1] = mask_color_red
            final_overlay_red = cv2.addWeighted(image_llava_np, 1 - alpha, overlay_image_red, alpha, 0)
            model_name = self.pretrained_model_name_or_path.split("/")[-1]
            output_image_path = output_dir / f"{model_name}_{image_path.name}_red.png"
            cv2.imwrite(output_image_path, cv2.cvtColor(final_overlay_red, cv2.COLOR_RGB2BGR))
            logger.debug("Saved red overlay image to %s", output_image_path)

            # Cyan overlay
            overlay_image_cyan = image_llava_np.copy()
            overlay_image_cyan[binary_mask == 1] = mask_color_cyan
            final_overlay_cyan = cv2.addWeighted(image_llava_np, 1 - alpha, overlay_image_cyan, alpha, 0)
            output_image_path = output_dir / f"{model_name}_{image_path.name}_cyan.png"
            cv2.imwrite(output_image_path, cv2.cvtColor(final_overlay_cyan, cv2.COLOR_RGB2BGR))
            logger.debug("Saved cyan overlay image to %s", output_image_path)

    def post_process(  # noqa:D102
        self,
        output_ids: torch.Tensor,
        pred_masks: torch.Tensor,
        pred_contact_3d: torch.Tensor,
        contact_type: Literal["hcontact", "hcontact-wScene", "oafford", "hcontact-ocontact", "h2dcontact"],
        image_path: Path,
        output_dir: Path,
        valid_masks_region: list[np.ndarray],
        overlay_sam_path: list[Path],
    ) -> None:
        pred_masks = pred_masks[0].detach().cpu().numpy()
        if pred_masks.shape[0] == 0:
            return

        # Save 3D contact vertices
        logger.debug("---> Num of non-zero contact vertices: %s", pred_contact_3d[pred_contact_3d != 0].shape[0])

        if contact_type == "hcontact":
            pred_contact_3d_smplx = convert_contacts(pred_contact_3d, self.smpl_to_smlpx_mapping)
            np.savez(
                output_dir / f"{image_path.stem}_hcontact_vertices.npz",
                pred_contact_3d_smplh=pred_contact_3d.cpu(),
                pred_contact_3d_smplx=pred_contact_3d_smplx.cpu(),
            )

            # Process SMPLX mesh with contact vertices
            output_smplx_path = output_dir / f"{image_path.stem}_smplx_body_with_hcontacts.obj"
            process_smplx_mesh_with_contacts(
                pred_contact_3d_smplx, output_smplx_path, contact_threshold=0.3, gender="neutral"
            )
        else:
            np.savez(output_dir / f"{image_path.stem}_oafford_vertices.npz", pred_contact_3d=pred_contact_3d.cpu())

            # Process object mesh with contact vertices for ocontact/oafford
            obj_mesh_path = image_path.with_name("object_mesh.obj")

            if obj_mesh_path.exists():
                output_obj_path = output_dir / f"{image_path.stem}_object_mesh_with_contacts_{contact_type}.obj"
                process_object_mesh_with_contacts(
                    obj_mesh_path, pred_contact_3d[0], output_obj_path, contact_threshold=0.5
                )
            else:
                logger.warning("object mesh not found at %s", obj_mesh_path)

        # Decode the output text
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        logger.debug("---> %s: %s", image_path.name, text_output)

        overlay_images = []
        for i, pred_mask in enumerate(pred_masks):
            pred_mask_binary = pred_mask > 0.3 if contact_type == "hcontact" else pred_mask > 0.5

            overlay_sam = cv2.imread(overlay_sam_path[i])
            overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_BGR2RGB)

            valid_mask_region = valid_masks_region[i]
            pred_mask_binary = np.logical_and(pred_mask_binary, valid_mask_region)

            # Expand pred_mask to match the RGB channels
            pred_mask_3d = np.stack([pred_mask_binary] * 3, axis=2)

            # Apply the mask and ensure result is uint8
            mask_color = (np.array([1.0, 0.15, 0.10]) * 255).astype(np.uint8)
            overlay_sam = np.where(pred_mask_3d, overlay_sam * 0.5 + mask_color * 0.5, overlay_sam)
            overlay_sam = np.clip(overlay_sam, 0, 255).astype(np.uint8)

            overlay_sam = cv2.cvtColor(overlay_sam, cv2.COLOR_RGB2BGR)

            # Store the overlay image
            overlay_images.append(overlay_sam)

        # Save the overlay images
        ## 2x2 grid
        h, w = overlay_images[0].shape[:2]
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        grid[:h, :w] = overlay_images[0]  # top-left
        grid[:h, w:] = overlay_images[1]  # top-right
        grid[h:, :w] = overlay_images[2]  # bottom-left
        grid[h:, w:] = overlay_images[3]  # bottom-right
        ## save
        model_name = self.pretrained_model_name_or_path.split("/")[-1]
        concat_save_path = output_dir / f"{model_name}_{image_path.stem}_{contact_type}_concat.jpg"
        cv2.imwrite(concat_save_path, grid)
        logger.debug("Concatenated image saved at: %s", concat_save_path)
        if image_path != output_dir / image_path.name:
            shutil.copy(image_path, output_dir)
