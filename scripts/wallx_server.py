# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:

Customized example:
```shell # port 33001-34000
python scripts/policy_server.py \
     --host=localhost \
     --port=33019 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch
from PIL import Image

from lerobot.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import SUPPORTED_POLICIES
from lerobot.scripts.server.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR
from wall_x.data.config import X2RDataProcessingConfig
from wall_x.data.utils import get_wallx_normal_text, process_grounding_points, replace_action_token, preprocesser_call

import os
import yaml
import torch
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from safetensors.torch import load_file
import torch.distributed.checkpoint as dcp
from transformers import AutoProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


# Camera key mappings for different datasets
CAMERA_KEY_MAPPINGS = {
    "lerobot/aloha_mobile_cabinet": {
        "observation.images.cam_high": "face_view",
        "observation.images.cam_left_wrist": "left_wrist_view",
        "observation.images.cam_right_wrist": "right_wrist_view",
    },
    "20250926/pickandplace": {
        "observation.images.top": "face_view",
        "observation.images.side": "wall_view",
    },
}


class VQAWrapper(object):
    def __init__(self, model_path: str, is_checkpoint: bool = False, pretrained_model_path: str = None):
        """
        Initialize VQA wrapper.
        
        Args:
            model_path (str): Path to model directory or checkpoint directory
            is_checkpoint (bool): If True, model_path is a checkpoint directory from accelerator.save_state
                                If False, model_path is a pretrained model directory
            pretrained_model_path (str): Required when is_checkpoint=True. Path to original pretrained model
                                       for loading processor and model architecture
        """
        self.device = self._setup_device()
        self.is_checkpoint = is_checkpoint
        self.pretrained_model_path = pretrained_model_path
        
        if is_checkpoint:
            if pretrained_model_path is None:
                raise ValueError("pretrained_model_path is required when loading from checkpoint")
            # For checkpoint loading, we need to load model weights from checkpoint 
            # and use the processor from the model
            self.model = self._load_model_from_checkpoint(model_path, pretrained_model_path)
            self.processor = self.model.processor
        else:
            # Standard pretrained model loading
            self.processor = self._load_processor(model_path)
            self.model = self._load_model(model_path)

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_processor(self, model_path: str) -> AutoProcessor:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _load_model(self, model_path: str) -> Qwen2_5_VLMoEForAction:
        model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        return model
    
    def _load_model_from_checkpoint(self, checkpoint_path: str, pretrained_model_path: str) -> Qwen2_5_VLMoEForAction:
        """
        Load model from FSDP2 checkpoint directory created by accelerator.save_state.
        
        Args:
            checkpoint_path (str): Path to checkpoint directory (e.g., "/path/to/save_path/epoch_79")
            pretrained_model_path (str): Path to original pretrained model for architecture
        
        Returns:
            Qwen2_5_VLMoEForAction: Loaded model
        """
        print(f"Loading model from FSDP2 checkpoint: {checkpoint_path}")
        
        # First, initialize the model architecture from pretrained path
        model = Qwen2_5_VLMoEForAction.from_pretrained(pretrained_model_path)
        model_weights_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(model_weights_path):
            print(f"Loading fallback weights from: {model_weights_path}")
            state_dict = load_file(model_weights_path, device="cpu")
            
            # Handle potential module prefix issues
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    new_key = key[7:]  # Remove 'module.' prefix
                elif key.startswith("_orig_mod."):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=False)

        model.to_bfloat16_for_selected_params()
        model.to(self.device)
        model.eval()
        
        return model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config

class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Hack: We fix the attributes here only for wallx
        self.device = 'cuda'
        self.policy_type = 'wallx'
        self.model_path = "/x2robot_v2/geoffrey/wall-x/workspace/checkpoints/99"
        self.pretrained_path = "/x2robot_v2/geoffrey/wall-x/wall-oss-flow"
        self.config_path = "/x2robot_v2/geoffrey/wall-x/workspace/lerobot_example/config_qact.yml"
        self.task_instruction = "Pick the white duck and place it in the pink cup"
        self.lerobot_features = None
        self.actions_per_chunk = 32
        self.policy = None
        
        # WallX preprocessing configuration
        self.data_config = None
        self.cam_key_mapping = None
        self.repo_id = None
        self.frame_index = 0  # Track frame index for observations
        self.min_stat = None
        self.delta_stat = None
        self.processor = None
        self.max_length = 768
        self.use_fast_tokenizer = False
        self.train_action_tokenizer = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        
        # Reset frame index for new episode
        self.frame_index = 0

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec
        self.lerobot_features = policy_specs.lerobot_features

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {self.policy_type} | "
            f"Actions per chunk: {self.actions_per_chunk} | "
            f"Device: {self.device}"
        )

        # self.lerobot_features = policy_specs.lerobot_features

        start = time.perf_counter()
        # load model from checkpoint
        wrapper = VQAWrapper(model_path=self.model_path, is_checkpoint=True, pretrained_model_path=self.pretrained_path)
        self.policy = wrapper.model
        self.policy.to(self.device)
        # load stats
        config = load_config(self.config_path)
        dataload_config = get_data_configs(config["data"])
        lerobot_config = dataload_config.get("lerobot_config", {})
        self.repo_id = lerobot_config.get("repo_id", "lerobot/aloha_mobile_cabinet")

        dataset_fps = 30

        delta_timestamps = {
            # action chunk
            "action": [
                t / dataset_fps
                for t in range(dataload_config.get("action_horizon", 32) - 1)
            ],
        }

        train_dataset = LeRobotDataset(
                self.repo_id,
                delta_timestamps=delta_timestamps,
                video_backend="pyav",
            )

        min_stat = train_dataset.meta.stats["action"]["min"]
        max_stat = train_dataset.meta.stats["action"]["max"]
        delta_stat = max_stat - min_stat
        print(f"min_stat: {min_stat}, max_stat: {max_stat}, delta_stat: {delta_stat}")
        self.min_stat = torch.tensor(min_stat).unsqueeze(0).to(self.device)
        self.delta_stat = torch.tensor(delta_stat).unsqueeze(0).to(self.device)
        
        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")
        
        # Initialize WallX preprocessing configuration
        self.cam_key_mapping = CAMERA_KEY_MAPPINGS.get(self.repo_id, {})
        
        self.data_config = X2RDataProcessingConfig().update(
            train_test_split=dataload_config["train_test_split"],
            split_seed=dataload_config["split_seed"],
            predict_action_keys=dataload_config["predict_action_keys"],
            obs_action_keys=dataload_config["obs_action_keys"],
            resolution=dataload_config.get("resolution", None),
            priority_order=dataload_config.get("priority_order", None),
        )
        
        # Load processor
        self.processor = wrapper.processor
        self.max_length = dataload_config.get("max_length", 768)
        self.use_fast_tokenizer = config.get("use_fast_tokenizer", False)
        
        # Load action tokenizer if using fast tokenizer
        if self.use_fast_tokenizer:
            action_tokenizer_path = config.get("action_tokenizer_path")
            if action_tokenizer_path:
                from transformers import AutoProcessor
                self.train_action_tokenizer = AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
        
        self.logger.info(f"WallX preprocessing initialized with repo_id: {self.repo_id}")
        self.logger.info(f"Resolution config: {self.data_config.resolution}")
        self.logger.info(f"Task instruction: {self.task_instruction}")
        self.logger.info(f"Max length: {self.max_length}")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.info(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        
        For WallX policy, this also applies full preprocessing similar to PreprocessedDataset:
        - Vision: Converts images to PIL format, applies resolution constraints, smart_resize
        - Text: Generates instruction text with camera views and action tokens
        - Action/State: Extracts agent position and prepares action format
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.device,
        )
        
        # Apply WallX-style preprocessing (vision + text + action)
        observation = self._apply_wallx_preprocessing(observation)
        
        # processed Observation - right keys, right dtype, right image shape, text, etc.
        return observation
    
    def _apply_wallx_preprocessing(self, observation: Observation) -> Observation:
        """
        Apply WallX-style preprocessing to observation.
        This matches the preprocessing done in PreprocessedDataset.__getitem__.
        
        Includes:
        - Vision preprocessing (PIL conversion, resolution constraints, smart_resize)
        - Text generation (instruction text with camera views and action tokens)
        - Agent position extraction
        
        Args:
            observation: Observation dict with image tensors
            
        Returns:
            Observation dict with preprocessed data ready for WallX inference
        """
        # 1. Vision preprocessing
        processed_images = []
        orig_height, orig_width = None, None
        resized_height, resized_width = None, None
        
        # Get image keys from the observation (filter for image observations)
        image_keys = [key for key in observation.keys() if 'image' in key.lower()]
        
        for key in image_keys:
            if key not in observation:
                continue
                
            # Get the image tensor (should be [C, H, W] in float32 [0,1] range)
            img_tensor = observation[key]
            
            # Convert from tensor to PIL image
            # Permute from [C, H, W] to [H, W, C] and convert to uint8 [0, 255]
            if img_tensor.ndim == 4:  # If batch dimension exists [B, C, H, W]
                img_tensor = img_tensor.squeeze(0)
            
            current_obs = img_tensor.clone().permute(1, 2, 0)
            img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
            orig_width, orig_height = img_pil.size
            
            # Apply resolution constraints (if config is not -1)
            # Map the observation key to the camera view name
            cam_view = self.cam_key_mapping.get(key, None)
            if cam_view:
                target_size = self.data_config.resolution.get(cam_view, -1)
                
                if target_size != -1:
                    # Maintain aspect ratio logic
                    if orig_width > orig_height:  # Landscape image
                        new_width = target_size
                        new_height = int(target_size * orig_height / orig_width)
                    else:  # Portrait image
                        new_height = target_size
                        new_width = int(target_size * orig_width / orig_height)
                    img_pil = img_pil.resize((new_width, new_height))
            
            # Apply smart scaling (qwen logic)
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.data_config.image_factor,
                min_pixels=self.data_config.min_pixels,
                max_pixels=self.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            processed_images.append(resized_img)
            
            # Update the observation with the PIL image
            observation[key] = resized_img
        
        # 2. Text preprocessing - generate instruction text
        if self.task_instruction is not None and orig_height is not None:
            instruction_info = {"instruction": self.task_instruction}
            action_chunk_size = self.actions_per_chunk - 1  # 32
            
            # Generate the complete text with instruction and action tokens
            complete_text, generate_subtask = get_wallx_normal_text(
                instruction_info,
                action_chunk_size,
                self.frame_index,
                self.data_config.priority_order,
                self.cam_key_mapping,
                generate_subtask_ratio=self.data_config.generate_subtask_ratio,
            )
            
            # Process grounding points (adjust coordinates for resized images)
            text = process_grounding_points(
                complete_text, 
                orig_height, 
                orig_width, 
                resized_height, 
                resized_width, 
                self.data_config.model_type
            )
            
            observation["text"] = text
            observation["image_inputs"] = processed_images
        
        # 3. Extract agent position (observation.state)
        # Look for state keys in observation
        for key in observation.keys():
            if 'state' in key.lower():
                observation["agent_pos"] = observation[key]
                break
        
        # 4. Track frame index
        observation["frame_index"] = self.frame_index
        self.frame_index += 1
        
        # 5. Apply DataCollator-style processing (normalization, tokenization, etc.)
        observation = self._apply_datacollator_processing(observation)
        
        return observation
    
    def _normalize(self, data: torch.Tensor, min_stat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Normalize data to [-1, 1] range"""
        return (data - min_stat) / delta * 2.0 - 1.0

    def _unnormalize(self, data: torch.Tensor, min_stat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Unnormalize data from [-1, 1] range"""
        return (data + 1.0) / 2.0 * delta + min_stat
    
    def _apply_datacollator_processing(self, observation: Observation) -> Observation:
        """
        Apply DataCollator-style processing to observation.
        This matches the processing done in DataCollator.__call__.
        
        Includes:
        - Agent position normalization and padding to 20 dims
        - Action normalization and padding (create dummy action for inference)
        - Action token replacement in text
        - Tokenization and image processing via preprocesser_call
        - MOE token type generation
        
        Args:
            observation: Observation dict with preprocessed data
            
        Returns:
            Observation dict ready for model inference
        """
        additional_inputs = {}
        
        # Convert min_stat and delta_stat to tensors
        min_stat = torch.tensor(self.min_stat, dtype=torch.float32)
        delta_stat = torch.tensor(self.delta_stat, dtype=torch.float32)
        
        # 1. Process agent position (proprioception)
        if "agent_pos" in observation:
            agent_pos = observation["agent_pos"]
            
            # Ensure proper shape [1, dim] for single observation
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            if agent_pos.dim() == 2:
                agent_pos = agent_pos.unsqueeze(1)  # [batch, 1, dim]
            
            # Handle NaN values
            agent_pos_mask = (~torch.isnan(agent_pos)).float()
            agent_pos = torch.nan_to_num(agent_pos, nan=0.0)
            
            # Normalize
            agent_pos = self._normalize(agent_pos, min_stat, delta_stat)
            
            # Pad to 20 dimensions if needed
            if agent_pos.shape[-1] != 20:
                padding_size = 20 - agent_pos.shape[-1]
                agent_pos = torch.cat(
                    [
                        agent_pos,
                        torch.zeros(agent_pos.shape[0], agent_pos.shape[1], padding_size)
                    ],
                    dim=-1
                )
                agent_pos_mask = torch.cat(
                    [
                        agent_pos_mask,
                        torch.zeros(agent_pos_mask.shape[0], agent_pos_mask.shape[1], padding_size)
                    ],
                    dim=-1
                )
            
            additional_inputs["proprioception"] = agent_pos
            additional_inputs["agent_pos_mask"] = agent_pos_mask
        
        # 2. Create dummy action chunk for inference (will be predicted by model)
        # Create dummy normalized action with proper shape [1, 1, action_dim]
        action_dim = len(min_stat)
        action = torch.zeros(1, 1, action_dim, dtype=torch.float32)
        dof_mask = torch.ones(1, 1, action_dim, dtype=torch.float32)
        
        # Pad to 20 dimensions if needed
        if action.shape[-1] != 20:
            padding_size = 20 - action.shape[-1]
            action = torch.cat(
                [
                    action,
                    torch.zeros(action.shape[0], action.shape[1], padding_size)
                ],
                dim=-1
            )
            dof_mask = torch.cat(
                [
                    dof_mask,
                    torch.zeros(dof_mask.shape[0], dof_mask.shape[1], padding_size)
                ],
                dim=-1
            )
        
        additional_inputs["action_chunk"] = action
        additional_inputs["dof_mask"] = dof_mask
        
        # 3. Get image inputs and text
        if "image_inputs" in observation:
            additional_inputs["image_inputs"] = [observation["image_inputs"]]
        
        if "text" in observation:
            additional_inputs["text"] = [observation["text"]]
        
        if "frame_index" in observation:
            additional_inputs["frame_index"] = torch.tensor([observation["frame_index"]], dtype=torch.long)
        
        # 4. Replace action tokens in text with dummy actions
        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            ["x2_normal"] * len(additional_inputs["text"]),
            additional_inputs["dof_mask"],
        )
        
        # 5. Process images and text with processor
        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        
        # 6. Create MOE token types (identify action tokens)
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id
        
        # 7. Merge all inputs
        inputs.update(additional_inputs)
        
        # 8. Add dataset name
        inputs["dataset_names"] = ["x2_normal"]
        
        return inputs

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        observation = observation.to(self.device)
        with torch.no_grad():
            chunk = self.policy(
                    **observation,
                    action_dim=20,
                    pred_horizon=self.actions_per_chunk,
                    mode="predict",
                    predict_mode="diffusion",
            )
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        chunk = chunk[:, : self.actions_per_chunk, :6]
        # unormalize action chunk
        chunk = self._unnormalize(chunk, self.min_stat, self.delta_stat)
        return chunk

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        preprocessing_time = time.perf_counter() - start_time

        self.last_processed_obs: TimedObservation = observation_t

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms | "
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms | "
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms | "
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
