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
        
        # # Check if this is an FSDP2 checkpoint (has pytorch_model_fsdp_0 directory)
        # fsdp_checkpoint_dir = os.path.join(checkpoint_path, "pytorch_model_fsdp_0")
        
        # if os.path.exists(fsdp_checkpoint_dir):
        #     print(f"Loading FSDP2 distributed checkpoint from: {fsdp_checkpoint_dir}")
            
        #     # Load the distributed checkpoint using torch.distributed.checkpoint
        #     try:
        #         # For single-process inference, we need to load the checkpoint differently
        #         # Create a state dict container
        #         state_dict = {"model": model.state_dict()}
                
        #         # Load the distributed checkpoint
        #         dcp.load(
        #             state_dict=state_dict,
        #             checkpoint_id=fsdp_checkpoint_dir,
        #         )
                
        #         # Extract the model state dict and load it
        #         loaded_state_dict = state_dict["model"]
                
        #         # Handle potential key mismatches by trying different approaches
        #         try:
        #             missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
                    
        #             if missing_keys:
        #                 print(f"Warning: Missing keys in checkpoint: {missing_keys[:10]}...")  # Show first 10
        #             if unexpected_keys:
        #                 print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}...")  # Show first 10
                        
        #             print("Successfully loaded FSDP2 checkpoint")
                    
        #         except Exception as load_error:
        #             print(f"Error loading state dict: {load_error}")
        #             # Try to match keys by removing prefixes
        #             cleaned_state_dict = {}
        #             for key, value in loaded_state_dict.items():
        #                 # Remove common FSDP prefixes
        #                 clean_key = key
        #                 if key.startswith("_fsdp_wrapped_module."):
        #                     clean_key = key[len("_fsdp_wrapped_module."):]
        #                 elif key.startswith("module."):
        #                     clean_key = key[len("module."):]
        #                 elif key.startswith("_orig_mod."):
        #                     clean_key = key[len("_orig_mod."):]
                        
        #                 cleaned_state_dict[clean_key] = value
                    
        #             missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        #             print(f"Loaded with key cleaning - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        #             print("Successfully loaded FSDP2 checkpoint with key cleaning")
                
        #     except Exception as e:
        #         print(f"Error loading FSDP2 checkpoint: {e}")
        #         print("Falling back to regular checkpoint loading...")
                
        # Fallback: try to load model.safetensors if available
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
        #         else:
        #             print(f"No fallback weights found. Available files: {os.listdir(checkpoint_path)}")
        # else:
        #     print(f"No FSDP2 checkpoint found at {fsdp_checkpoint_dir}")
        #     print(f"Available directories: {os.listdir(checkpoint_path)}")
        
        # Move model to device and set to eval mode
        # if self.device == "cuda":
        #     model = model.to_bfloat16_for_selected_params()
        # else:
        #     model.to(self.device)
        model.to_bfloat16_for_selected_params()
        model.to(self.device)
        model.eval()
        
        return model

save_dir = "/x2robot_v2/geoffrey/wall-x/workspace/openloop_plot"
wrapper = VQAWrapper(model_path="/x2robot_v2/geoffrey/wall-x/workspace/checkpoints/99", is_checkpoint=True, pretrained_model_path="/x2robot_v2/geoffrey/wall-x/wall-oss-flow")
model = wrapper.model

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


# get test dataloader
path = "/x2robot_v2/geoffrey/wall-x/workspace/lerobot_example/config_qact.yml"
config = load_config(path)
dataload_config = get_data_configs(config["data"])
lerobot_config = dataload_config.get("lerobot_config", {})
repo_id = lerobot_config.get("repo_id", "lerobot/aloha_mobile_cabinet")

dataset_fps = 30
dataload_config = get_data_configs(config["data"])

delta_timestamps = {
    # action chunk
    "action": [
        t / dataset_fps
        for t in range(dataload_config.get("action_horizon", 32) - 1)
    ],
}

train_dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

min_stat = train_dataset.meta.stats["action"]["min"]
max_stat = train_dataset.meta.stats["action"]["max"]
delta_stat = max_stat - min_stat
print(f"min_stat: {min_stat}, max_stat: {max_stat}, delta_stat: {delta_stat}")


dataset = load_test_dataset(config, train_dataset.meta, lerobot_config, seed=19)
dataloader = dataset.get_dataloader()


total_frames = len(dataloader)

pred_horizon = 32
action_dim = 6
gt_traj = torch.zeros((total_frames, action_dim))
pred_traj = torch.zeros((total_frames, action_dim))

for idx, batch in enumerate(dataloader):
    if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
        batch = batch.to("cuda")
        print(batch.keys())
        with torch.no_grad():
            outputs = model(
                **batch,
                action_dim=20,
                pred_horizon=pred_horizon,
                mode="predict",
                predict_mode="diffusion",
            )
        pred_traj[idx : idx + pred_horizon] = outputs["predict_action"].detach().cpu()[:, :, :action_dim]

        # Denormalize ground truth actions
        gt_action_chunk = batch["action_chunk"][:, :, :action_dim]
        dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
        # denormalized_gt = model.action_preprocessor.normalizer_action.unnormalize_data(
        #     gt_action_chunk, ["x2_normal"], dof_mask
        # )
        # gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()
        gt_traj[idx : idx + pred_horizon] = gt_action_chunk.detach().cpu()

gt_traj = torch.clamp(gt_traj, -1, 1)
pred_traj = torch.clamp(pred_traj, -1, 1)
gt_traj = (gt_traj + 1) / 2
pred_traj = (pred_traj + 1) / 2
gt_traj = gt_traj * torch.tensor(delta_stat).unsqueeze(0) + torch.tensor(min_stat).unsqueeze(0) 
pred_traj = pred_traj * torch.tensor(delta_stat).unsqueeze(0) + torch.tensor(min_stat).unsqueeze(0)

gt_traj_np = gt_traj.numpy()
pred_traj_np = pred_traj.numpy()

timesteps = gt_traj.shape[0]

fig, axs = plt.subplots(action_dim, 1, figsize=(15, 5 * action_dim), sharex=True)
fig.suptitle("Action Comparison for lerobot", fontsize=16)

for i in range(action_dim):
    axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
    axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
    axs[i].set_ylabel(f"Action Dim {i+1}")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Timestep")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "lerobot_comparison.png"))
plt.close()
