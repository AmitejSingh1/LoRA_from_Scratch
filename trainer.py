import torch
import torch.nn as nn
from typing import Dict

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts only the LoRA parameters from a model.
    Instead of a 10GB checkpoint for a full LLM, this isolates the ~20MB of LoRA weights!
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # We ONLY save the trainable parameters (our A and B matrices)
            lora_state_dict[name] = param.data.cpu()
    return lora_state_dict

def save_lora_weights(model: nn.Module, save_path: str):
    """Saves the extracted LoRA weights to disk."""
    state_dict = get_lora_state_dict(model)
    torch.save(state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}.")
    print(f"Total LoRA parameters saved: {sum(p.numel() for p in state_dict.values()):,}")

def load_lora_weights(model: nn.Module, load_path: str):
    """
    Loads LoRA weights back into the model.
    Must be called AFTER the model has been injected with `inject_lora`.
    """
    state_dict = torch.load(load_path, map_location="cpu")
    # `strict=False` is mathematically required because our loaded state_dict 
    # ONLY contains LoRA weights, not the base model's millions of frozen weights.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded LoRA weights from {load_path}.")
    if unexpected_keys:
        print(f"Warning - unexpected keys found: {unexpected_keys}")

def train_lora_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Runs one epoch of training using Mixed Precision (AMP).
    Given an 8GB VRAM GPU (like the RTX 4060), Mixed Precision enables us 
    to fit larger batch sizes by calculating gradients in Float16 instead of Float32.
    """
    model.train()
    total_loss = 0.0
    
    # Creates a GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    model.to(device)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Zero out old gradients
        optimizer.zero_grad()
        
        # 2. Mixed precision context manager
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            # 3. Scale the loss and call backward (prevents float16 underflow)
            scaler.scale(loss).backward()
            
            # 4. Unscale the gradients and update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Fallback for CPU training (just standard PyTorch)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Complete. Average Loss: {avg_loss:.4f}")
    return avg_loss
