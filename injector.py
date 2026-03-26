import torch
import torch.nn as nn
from lora import LoRALinear, LoRAConv2d

def inject_lora(
    model: nn.Module, 
    target_layer_names: list[str], 
    r: int = 8, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.05
) -> nn.Module:
    """
    Recursively replaces target nn.Linear or nn.Conv2d layers in a model with LoRA adapters.
    """
    # 1. Freeze EVERYTHING in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Track how many layers we successfully swap
    injected_count = 0
    
    def replace_modules(parent_module):
        nonlocal injected_count
        
        for name, child in parent_module.named_children():
            # Check if it matches our targeting string
            is_target = ("all" in target_layer_names) or any(target in name for target in target_layer_names)
            
            if is_target:
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(parent_module, name, lora_layer)
                    injected_count += 1
                elif isinstance(child, nn.Conv2d):
                    lora_layer = LoRAConv2d(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(parent_module, name, lora_layer)
                    injected_count += 1
                else:
                    # Target name matched, but it's not a Linear or Conv2d layer
                    replace_modules(child)
            else:
                replace_modules(child)
                
    # Kick off the recursion
    replace_modules(model)
    print(f"Successfully injected {injected_count} LoRA layers.")
    
    return model

if __name__ == "__main__":
    import sys
    import os
    
    # Let's test it against your Prostate VGG-UNet model!
    sys.path.append("C:/personal_proj/prostate")
    from model import build_vgg16_unet
    
    print("Loading Base Prostate VGG-UNet...")
    # Because of torchvision weights, we'll silence PyTorch warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    dummy_model = build_vgg16_unet()
    
    print("\n--- Injecting LoRA into Convolutions ---")
    # We will target all layers containing "conv" or "up" in the U-Net
    lora_model = inject_lora(dummy_model, target_layer_names=["conv", "up", "dec", "enc"], r=8)
    
    # Prove the memory savings on a massive Vision Model!
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)
    
    print(f"\n--- U-NET PARAMETER COUNT ---")
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
    print(f"Frozen parameters (Base):    {frozen_params:,}")
    print(f"% Trainable:                 {100 * trainable_params / (trainable_params + frozen_params):.4f}%")
