import torch
import os
from PIL import Image
from transformers import AutoProcessor
from safetensors.torch import load_file
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


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
        
        # Check if this is an FSDP2 checkpoint (has pytorch_model_fsdp_0 directory)
        fsdp_checkpoint_dir = os.path.join(checkpoint_path, "pytorch_model_fsdp_0")
        
        if os.path.exists(fsdp_checkpoint_dir):
            print(f"Loading FSDP2 distributed checkpoint from: {fsdp_checkpoint_dir}")
            
            # Load the distributed checkpoint using torch.distributed.checkpoint
            try:
                # For single-process inference, we need to load the checkpoint differently
                # Create a state dict container
                state_dict = {"model": model.state_dict()}
                
                # Load the distributed checkpoint
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=fsdp_checkpoint_dir,
                )
                
                # Extract the model state dict and load it
                loaded_state_dict = state_dict["model"]
                
                # Handle potential key mismatches by trying different approaches
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"Warning: Missing keys in checkpoint: {missing_keys[:10]}...")  # Show first 10
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}...")  # Show first 10
                        
                    print("Successfully loaded FSDP2 checkpoint")
                    
                except Exception as load_error:
                    print(f"Error loading state dict: {load_error}")
                    # Try to match keys by removing prefixes
                    cleaned_state_dict = {}
                    for key, value in loaded_state_dict.items():
                        # Remove common FSDP prefixes
                        clean_key = key
                        if key.startswith("_fsdp_wrapped_module."):
                            clean_key = key[len("_fsdp_wrapped_module."):]
                        elif key.startswith("module."):
                            clean_key = key[len("module."):]
                        elif key.startswith("_orig_mod."):
                            clean_key = key[len("_orig_mod."):]
                        
                        cleaned_state_dict[clean_key] = value
                    
                    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                    print(f"Loaded with key cleaning - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                    print("Successfully loaded FSDP2 checkpoint with key cleaning")
                
            except Exception as e:
                print(f"Error loading FSDP2 checkpoint: {e}")
                print("Falling back to regular checkpoint loading...")
                
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
                else:
                    print(f"No fallback weights found. Available files: {os.listdir(checkpoint_path)}")
        else:
            print(f"No FSDP2 checkpoint found at {fsdp_checkpoint_dir}")
            print(f"Available directories: {os.listdir(checkpoint_path)}")
        
        # Move model to device and set to eval mode
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        
        return model

    def generate(self, image: Image.Image, text: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_params = {
            "max_new_tokens": 1024,  # default value, can be overridden by kwargs
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


if __name__ == "__main__":
    # Example 1: Load from pretrained model directory
    # MODEL_PATH_FOR_MODULE_TEST = "/x2robot_v2/geoffrey/wall-x/wall-oss-flow"
    # wrapper = VQAWrapper(model_path=MODEL_PATH_FOR_MODULE_TEST, is_checkpoint=False)
    
    # Example 2: Load from FSDP2 checkpoint directory (saved by accelerator.save_state with FSDP2=True)
    CHECKPOINT_PATH = "/x2robot_v2/geoffrey/wall-x/workspace/checkpoints_/49"  # epoch 49 checkpoint
    PRETRAINED_PATH = "/x2robot_v2/geoffrey/wall-x/wall-oss-flow"  # Original pretrained model path
    wrapper = VQAWrapper(
        model_path=CHECKPOINT_PATH, 
        is_checkpoint=True, 
        pretrained_model_path=PRETRAINED_PATH
    )

    try:
        test_question = "To move the red block in the plate with same color, what should you do next? Think step by step."

        # Local Image
        img = Image.open("/x2robot_v2/geoffrey/wall-x/assets/cot_example_frame.png").convert("RGB")
        # Internet Image
        # import requests
        # test_image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        # img = Image.open(requests.get(test_image_url, stream=True).raw).convert("RGB")

        answer = wrapper.generate(img, test_question)

        print("model answer:", answer)
    except Exception as e:
        print(f"model answer fail: {e}")
