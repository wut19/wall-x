"""
Data processing utilities for Wall-X multimodal robotic learning.

This module provides utilities for preprocessing text, images, and actions
for multimodal transformer models in robotic learning tasks.
"""

import re
import torch
import random
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import BatchFeature
from dataclasses import dataclass
import json

KEY_MAPPINGS = {
    "lerobot/aloha_mobile_cabinet": {
        "camera":{
            "observation.images.cam_high": "face_view",
            "observation.images.cam_left_wrist": "left_wrist_view",
            "observation.images.cam_right_wrist": "right_wrist_view",
        },
        "state": "observation.state",
        "action": "action"
    },
    "physical-intelligence/libero":{
        "camera":{
            "image" : "face_view",
            "wrist_image" : "left_wrist_view",
        },
        "state": "state",
        "action": "actions"
    }
}

CAMERA_NAME_MAPPING = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
    "move1_view": "move view",
    "move2_view": "move view",
    "wall_view": "wall view",
    "top_view": "top view",
}


MULTIMODAL_DATASET_NAMES = [
    "x2_multimodal_from_action",
    "x2_multimodal",
    "x2_subtask_generation",
    "multimodal_CapsFusion",
    "multimodal_Robo2VLM",
    "multimodal_RoboPoint",
    "multimodal_EQA",
    "multimodal_Cambrian",
    "multimodal_pixmo",
    "multimodal_VQAv2",
    "multimodal_COCO",
]

FREQUENCY_MAPPING = {
    "x2_normal": 32,
    "fractal": 5,
    "bridge_data_v2": 5,
    "droid": 15,
    "agibotworld_alpha": 32,
    "DobbE": 30,
    "RH20T": 10,
    "UMI-biarm": 10,
    "austin_buds": 20,
    "austin_sailor": 20,
    "austin_sirius": 20,
    "bc_z": 10,
    "berkeley_autolab_ur5": 5,
    "berkeley_cable_routing": 10,
    "berkeley_fanuc_manipulation": 10,
    "dlr_edan_shared_control": 5,
    "fmb": 10,
    "furniture_bench": 10,
    "jaco_play": 10,
    "nyu_rot": 10,
    "stanford_hydra": 10,
    "stanford_kuka_multimodal": 20,
    "taco_play": 30,
    "utaustin_mutex": 20,
    "viola": 20,
}


def preprocesser_call(
    processor,
    images: Optional[Union[List, Any]] = None,
    text: Optional[Union[str, List[str]]] = None,
    videos: Optional[Union[List, Any]] = None,
    padding: Union[bool, str] = False,
    truncation: Optional[bool] = None,
    max_length: Optional[int] = None,
    return_tensors: str = "pt",
) -> BatchFeature:
    """Unified preprocessing function for Wall-X model handling text, image and video inputs.

    Processes inputs into format suitable for multimodal transformer models, including:
    - Text tokenization and special token handling
    - Image/video processing through image processor
    - Attention mask and label generation
    - Padding and truncation handling

    Args:
        processor: Multimodal processor containing tokenizer and image processor
        images: Input images (PIL, numpy arrays, or torch tensors)
        text: Text or list of texts to tokenize
        videos: Input videos (numpy arrays or torch tensors)
        padding: Whether to pad sequences to same length
        truncation: Whether to truncate sequences longer than max_length
        max_length: Maximum length for truncation/padding
        return_tensors: Format for returned tensors ('pt', 'np', etc.)

    Returns:
        BatchFeature containing processed inputs with keys:
        - input_ids: Tokenized text
        - attention_mask: Attention mask for text
        - pixel_values: Processed image pixels
        - pixel_values_videos: Processed video frames
        - image_grid_thw: Image grid dimensions for LLM
        - video_grid_thw: Video grid dimensions for LLM
        - labels: Training labels with masking
    """
    # Process image inputs
    if images is not None and len(images) > 0:
        image_inputs = processor.image_processor(
            images=images, videos=None, return_tensors=return_tensors
        )
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        image_inputs = {}
        image_grid_thw = None

    # Process video inputs
    if videos is not None:
        videos_inputs = processor.image_processor(
            images=None, videos=videos, return_tensors=return_tensors
        )
        video_grid_thw = videos_inputs["video_grid_thw"]
    else:
        videos_inputs = {}
        video_grid_thw = None

    # Ensure text input is in list format
    if not isinstance(text, list):
        text = [text]

    # Process image placeholder tokens in text
    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|image_pad|>" in text[i]:
                # Add bounds checking to avoid index overflow
                if index >= len(image_grid_thw):
                    print(
                        f"Warning: Number of image placeholders ({index + 1}) "
                        f"exceeds actual images ({len(image_grid_thw)}), "
                        f"skipping remaining placeholder processing"
                    )
                    break
                # Replace image placeholder with actual token count
                token_count = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    "<|image_pad|>", "<|placeholder|>" * token_count, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

    # Process video placeholder tokens in text
    if video_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|video_pad|>" in text[i]:
                # Replace video placeholder with actual token count
                token_count = video_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    "<|video_pad|>", "<|placeholder|>" * token_count, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

    # Tokenize complete input text
    text_inputs = processor.tokenizer(
        text,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    # Get pad token ID for label generation
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    # Generate labels for multi-turn dialogue, keeping only assistant response loss
    labels = torch.full_like(text_inputs.input_ids, -100)
    assistant_marker = "<|im_start|>assistant\n"
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_tokens = processor.tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    ).input_ids

    for i in range(len(text)):
        assistant_regions = []
        parts = text[i].split(assistant_marker)

        # Process each part to determine which tokens belong to assistant responses
        # Count left padding tokens
        num_left_pads = 0
        for token_id in text_inputs.input_ids[i]:
            if token_id == pad_token_id:
                num_left_pads += 1
            else:
                break
        current_pos = num_left_pads

        for j, part in enumerate(parts):
            part_tokens = processor.tokenizer(part, add_special_tokens=False).input_ids
            if j == 0:
                # First part is system prompt or user question, all labels are -100
                current_pos += len(part_tokens)
                continue

            # From second part onwards, each part starts with assistant response
            for k in range(current_pos + 1, len(text_inputs.input_ids[i])):
                if text_inputs.input_ids[i][k] == im_end_token_id:
                    assistant_regions.append(
                        (current_pos + len(assistant_tokens), k + 2)
                    )
                    break
            current_pos += len(part_tokens) + 3

        # Set labels for assistant response regions
        for start, end in assistant_regions:
            labels[i][start:end] = text_inputs.input_ids[i][start:end]

    # Mask special action tokens in labels
    action_token_id = processor.tokenizer.encode("<|action|>")[0]
    propri_token_id = processor.tokenizer.encode("<|propri|>")[0]
    labels[labels == action_token_id] = -100
    labels[labels == propri_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Set labels to None if all are invalid to skip cross entropy loss
    if (labels != -100).any().item():
        text_inputs["labels"] = labels
    else:
        text_inputs["labels"] = None

    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})


def process_grounding_points(
    text: str,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    model_type: str,
) -> str:
    """Process grounding point coordinates in text based on image resizing.

    Adjusts coordinate values in <point> tags to match resized image dimensions
    for different model types (qwen2, qwen2_5).

    Args:
        text: Input text containing <point> tags with coordinates
        orig_height: Original image height
        orig_width: Original image width
        resized_height: Resized image height
        resized_width: Resized image width
        model_type: Model type for coordinate processing ('qwen2' or 'qwen2_5')

    Returns:
        Text with adjusted coordinate values
    """
    # Regex pattern to match <point> tags and their contents
    point_pattern = re.compile(r"<point>(.*?)</point>")

    def process_match(match):
        """Process a single point match and adjust coordinates."""
        coords_str = match.group(1)
        try:
            # Extract coordinates from string
            coords = list(map(int, re.findall(r"\d+", coords_str)))

            # Calculate resize scale factors
            scale_w = resized_width / orig_width
            scale_h = resized_height / orig_height

            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    # Qwen2.5 uses pixel coordinates
                    new_x = max(0, min(round(x * scale_w), resized_width - 1))
                    new_y = max(0, min(round(y * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    # Qwen2 normalizes to [0, 1000) range
                    new_x = max(0, min(999.999, (x / orig_width) * 1000))
                    new_y = max(0, min(999.999, (y / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x, new_y]

            elif len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    new_x1 = max(0, min(round(x1 * scale_w), resized_width - 1))
                    new_y1 = max(0, min(round(y1 * scale_h), resized_height - 1))
                    new_x2 = max(0, min(round(x2 * scale_w), resized_width - 1))
                    new_y2 = max(0, min(round(y2 * scale_h), resized_height - 1))
                elif model_type == "qwen2":
                    new_x1 = max(0, min(999.999, (x1 / orig_width) * 1000))
                    new_y1 = max(0, min(999.999, (y1 / orig_height) * 1000))
                    new_x2 = max(0, min(999.999, (x2 / orig_width) * 1000))
                    new_y2 = max(0, min(999.999, (y2 / orig_height) * 1000))
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                coords = [new_x1, new_y1, new_x2, new_y2]

            # Return processed point tag
            return f'<point>[{", ".join(map(str, coords))}]</point>'

        except (ValueError, TypeError):
            # Return original content if processing fails
            return match.group(0)

    # Replace all matching point tags
    processed_text = point_pattern.sub(process_match, text)
    return processed_text


def get_frame_instruction(
    instruction_info: Dict[str, Any],
    frame_idx: Optional[int] = None,
    truncate_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Optional[int]]:
    """Extract frame-specific instruction from instruction dictionary.

    Args:
        instruction_info: Dictionary containing instruction components
        frame_idx: Current frame index
        truncate_keys: Keys that trigger truncation when found

    Returns:
        Tuple of (frame_instruction_dict, split_end_frame)
    """
    if truncate_keys is None:
        truncate_keys = [
            "subtask_generation",
            "distribute",
            "subtask_generation_zh",
            "distribute_zh",
        ]

    instruction_for_frame = {}
    split_end = None

    for key, value in instruction_info.items():
        if isinstance(value, dict):
            # Handle frame-range specific instructions
            for frame_range, frame_instruction in value.items():
                start_frame, end_frame = map(int, frame_range.split(" "))
                if start_frame <= frame_idx < end_frame or (start_frame == frame_idx):
                    instruction_for_frame[key] = frame_instruction
                    if (
                        truncate_keys is not None
                        and split_end is None
                        and key in truncate_keys
                    ):
                        split_end = end_frame + 1
                    break
        else:
            instruction_for_frame[key] = value

    return instruction_for_frame, split_end


def get_task_instruction(
    frame_instruction_info: Dict[str, Any], priority_order: Optional[OrderedDict] = None
) -> str:
    """Construct task instruction from available instruction fields using priority sampling.

    Args:
        frame_instruction_info: Dictionary containing instruction fields
        priority_order: OrderedDict specifying sampling probability for each field

    Returns:
        Combined instruction string with priority components
    """
    # Default priority settings
    default_priority_order = OrderedDict(
        {
            "subtask_generation": 0.25,
            "subtask_generation_zh": 0.25,
            "distribute": 0.25,
            "distribute_zh": 0.25,
        }
    )

    if priority_order is not None:
        priority_order = OrderedDict(priority_order)
    else:
        priority_order = default_priority_order

    got_instruction = False
    task_instruction = ""

    # Sample instruction components based on priority probabilities
    for key, prob in priority_order.items():
        if key in frame_instruction_info and frame_instruction_info[key] != "":
            if got_instruction:
                if random.random() >= prob:
                    continue

            task_instruction += f"\n{frame_instruction_info[key]}"
            got_instruction = True
            break

    # Fall back to base instruction if no priority components found
    if not got_instruction:
        task_instruction = frame_instruction_info.get("instruction", "")

    return task_instruction


def get_wallx_normal_text(
    instruction_info: Dict[str, Any],
    action_chunk_size: int,
    frame_idx: int,
    priority_order: Optional[OrderedDict] = None,
    cam_mapping: Optional[Dict[str, str]] = None,
    generate_subtask_ratio: float = 0.0,
) -> Tuple[str, bool]:
    """Construct complete multimodal prompt text for Wall-X model.

    Formats input using special tokens including:
    - System message
    - User observations (with image placeholders)
    - Task instructions
    - Proprioception prompts
    - Assistant responses (with action tokens)

    Args:
        instruction_info: Dictionary containing instruction components
        action_chunk_size: Number of action tokens to generate
        frame_idx: Current frame index
        priority_order: Priority order for instruction sampling
        cam_mapping: Camera name mapping dictionary
        generate_subtask_ratio: Probability of generating subtask instead of actions

    Returns:
        Tuple of (formatted_prompt_text, is_subtask_generation)
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if cam_mapping:
        for _, cam_name in cam_mapping.items():
            view_name = CAMERA_NAME_MAPPING.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    # Get frame-specific instruction
    frame_instruction_info, _ = get_frame_instruction(
        instruction_info, frame_idx=frame_idx
    )

    generate_subtask = False
    priority_keys = ["subtask_generation", "distribute"]

    # Decide whether to generate subtask or actions
    if (
        bool(set(frame_instruction_info.keys()) & set(priority_keys))
        and random.random() < generate_subtask_ratio
    ):
        # Generate subtask (equivalent to VQA task)
        instruction = frame_instruction_info.get("instruction", "")
        text_prompt = "\nPredict the next action in language.\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"

        # Find output instruction from priority keys
        for key in priority_keys:
            if key in frame_instruction_info:
                output_instruction = frame_instruction_info[key]
                break

        assistant_output = (
            f"{role_start_symbol}assistant\n{output_instruction}\n{role_end_symbol}"
        )
        generate_subtask = True
    else:
        # Generate actions
        instruction = get_task_instruction(
            frame_instruction_info, priority_order=priority_order
        )
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_output = f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n{action_symbol * action_chunk_size}"

    complete_text = prologue + user_message + assistant_output
    return complete_text, generate_subtask


def get_action_tokens(
    normalized_actions: Union[torch.Tensor, List], action_tokenizer
) -> List[List[str]]:
    """Convert normalized actions to action token strings.

    Args:
        normalized_actions: Normalized action arrays/tensors
        action_tokenizer: Tokenizer for converting actions to tokens

    Returns:
        List of action token string lists for each sample
    """
    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    all_action_tokens = []
    for i in range(len(normalized_actions)):
        if isinstance(normalized_actions[i], torch.Tensor):
            normalized_actions[i] = normalized_actions[i].cpu().numpy()

        token_id = action_tokenizer(normalized_actions[i])
        action_tokens = [f"<|action_token_{j}|>" for j in token_id[0]]
        all_action_tokens.append(action_tokens)

    return all_action_tokens


def pad_action_token_strs(
    actions_token_lists: List[List[str]], pad_token: str = "<|endoftext|>"
) -> List[str]:
    """Pad action token lists to same length and join as strings.

    Args:
        actions_token_lists: List of action token lists for each sample
        pad_token: Token used for padding

    Returns:
        List of padded action token strings
    """
    max_len = max(len(tokens) for tokens in actions_token_lists)
    padded_action_strs = []

    for tokens in actions_token_lists:
        padded_tokens = (
            tokens + ["<|im_end|>\n"] + [pad_token] * (max_len - len(tokens))
        )
        padded_action_strs.append("".join(padded_tokens))

    return padded_action_strs


def replace_action_token(
    text: List[str],
    norm_action: Optional[torch.Tensor],
    action_tokenizer,
    dataset_names: List[str],
    dof_masks: Optional[torch.Tensor] = None,
) -> List[str]:
    """Replace action placeholders in text with actual action tokens.

    Args:
        text: List of text strings with action placeholders
        norm_action: Normalized action tensors
        action_tokenizer: Tokenizer for converting actions to tokens
        dataset_names: Names of datasets for each sample
        dof_masks: Masks for degrees of freedom

    Returns:
        List of text strings with action tokens replaced
    """
    # Filter out multimodal dataset names
    dataset_names = [
        name for name in dataset_names if name not in MULTIMODAL_DATASET_NAMES
    ]

    # Get required action chunk sizes
    required_chunk_sizes = [FREQUENCY_MAPPING.get(name, 32) for name in dataset_names]

    if action_tokenizer is not None and norm_action is not None:
        # Extract actions based on chunk sizes and DOF masks
        norm_action = [
            action[: required_chunk_sizes[i], dof_masks[i, 0].bool()]
            for i, action in enumerate(norm_action)
        ]

        # Convert to action tokens and pad
        actions_fast_tokens = get_action_tokens(norm_action, action_tokenizer)
        actions_fast_token_strs = pad_action_token_strs(actions_fast_tokens)

        # Replace action placeholders with actual tokens
        actions_fast_token_idx = 0
        for i in range(len(text)):
            if "<|action_fast|>" in text[i]:
                text[i] = text[i].replace(
                    "<|action_fast|><|im_end|>\n",
                    actions_fast_token_strs[actions_fast_token_idx],
                )
                actions_fast_token_idx += 1

        # Remove remaining action placeholders
        text = [t.replace("<|action|>", "") for t in text]
    else:
        # Remove action placeholders when no tokenizer available
        text = [t.replace("<|action_fast|><|im_end|>\n", "") for t in text]

    return text

@dataclass
class NormStats:
    min: torch.Tensor
    max: torch.Tensor
    delta: torch.Tensor

def load_norm_stats(norm_stats_path, dataset_name):
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    action_key = KEY_MAPPINGS[dataset_name]["action"]
    state_key = KEY_MAPPINGS[dataset_name]["state"]
    q01=torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    q99=torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    delta = q99 - q01
    action_norm_stats = NormStats(
        min=q01,
        max=q99,
        delta=delta,
    )
    q01=torch.tensor(norm_stats["norm_stats"][state_key]["q01"])
    q99=torch.tensor(norm_stats["norm_stats"][state_key]["q99"])
    delta = q99 - q01
    state_norm_stats = NormStats(
        min=q01,
        max=q99,
        delta=delta,
    )
    
    return {"action": action_norm_stats, "state": state_norm_stats}
