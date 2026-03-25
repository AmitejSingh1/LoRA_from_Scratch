import torch
import torch.nn as nn
from lora import LoRALinear

def inject_lora(
    model: nn.Module, 
    target_layer_names: list[str], 
    r: int = 8, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.05
) -> nn.Module:
    """
    Recursively replaces target nn.Linear layers in a model with LoRALinear adapters.
    
    Args:
        model: The base PyTorch model.
        target_layer_names: List of strings (e.g., ["q_proj", "v_proj"]) to target.
                            If a layer's name contains any of these targets, it gets replaced.
                            Pass ["all"] to replace every single nn.Linear layer.
    """
    # 1. Freeze EVERYTHING in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Track how many layers we successfully swap
    injected_count = 0
    
    # 2. We use a helper function for the recursion because replacing a module
    # requires setting the attribute on its parent module.
    def replace_modules(parent_module):
        nonlocal injected_count
        
        for name, child in parent_module.named_children():
            # Check if it's a linear layer and if it matches our targeting string
            is_linear = isinstance(child, nn.Linear)
            is_target = ("all" in target_layer_names) or any(target in name for target in target_layer_names)
            
            if is_linear and is_target:
                # Create the LoRA wrapper
                lora_layer = LoRALinear(
                    base_layer=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                
                # Rip out the original layer and replace it in the parent module
                setattr(parent_module, name, lora_layer)
                injected_count += 1
            else:
                # Not a target layer (e.g. it's a CNN layer, LayerNorm, or un-targeted Linear).
                # Recursively search its children
                replace_modules(child)
                
    # 3. Kick off the recursion
    replace_modules(model)
    print(f"Successfully injected {injected_count} LoRA layers.")
    
    # Return the model (note: mutation happens in-place)
    return model


if __name__ == "__main__":
    # --- QUICK TEST SCRIPT ---
    print("Testing the LoRA Dynamic Injector...")
    
    # Create a dummy model with nested linear layers
    dummy_model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(2048, 2048),  # Nested linear layer
            nn.ReLU()
        ),
        nn.Linear(2048, 10)
    )
    
    print("\nBefore Injection:")
    print(dummy_model)
    
    # Target every single linear layer
    lora_model = inject_lora(dummy_model, target_layer_names=["all"], r=8)
    
    print("\nAfter Injection:")
    print(lora_model)
    
    # Prove the memory savings!
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)
    
    print(f"\n--- PARAMETER COUNT ---")
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
    print(f"Frozen parameters (Base):    {frozen_params:,}")
    print(f"% Trainable:                 {100 * trainable_params / (trainable_params + frozen_params):.4f}%")
