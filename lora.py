import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    A custom PyTorch module that implements Low-Rank Adaptation (LoRA) for a linear layer.
    It wraps an existing nn.Linear layer and adds trainable low-rank matrices A and B.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        
        # 1. Store the original pre-trained layer
        self.base_layer = base_layer
        
        # Freeze the base layer's parameters so we don't update W_0
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # 2. LoRA Hyperparameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # 3. LoRA Matrices A and B
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Matrix A: maps from in_features down to rank r
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        # Matrix B: maps from rank r back to out_features
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        
        # Dropout layer to regularize LoRA
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 4. Initialize weights properly
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize A with Kaiming uniform (like a standard linear layer)
        Initialize B with zeros so that the initial forward pass is identical to the base model.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Base output + (Dropout(x) @ A^T @ B^T) * scaling
        """
        # Original frozen forward pass
        base_output = self.base_layer(x)
        
        # LoRA forward pass: x -> dropout -> A -> B -> scale
        # We use F.linear which performs: input @ weight.T
        lora_action = self.dropout(x)
        lora_action = F.linear(lora_action, self.lora_A) # Shape: (..., r)
        lora_action = F.linear(lora_action, self.lora_B) # Shape: (..., out_features)
        
        # Add the LoRA adapters to the base output
        return base_output + (lora_action * self.scaling)

    def merge_and_unload(self) -> nn.Linear:
        """
        Merges the LoRA weights into the base layer for zero-overhead inference
        and returns the standard nn.Linear layer.
        W_new = W_base + (B @ A) * scaling
        """
        # Calculate the weight delta
        # lora_B is (out, r), lora_A is (r, in) -> B @ A is (out, in)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        
        # Add delta directly to the base layer's weights
        self.base_layer.weight.data += delta_w
        
        # Unfreeze if necessary (optional depending on future training needs)
        for param in self.base_layer.parameters():
            param.requires_grad = True
            
        return self.base_layer

# Quick test to show it works
if __name__ == "__main__":
    print("Testing LoRALinear Initialization...")
    
    # 1. Create a dummy base linear layer (e.g., from an LLM)
    base = nn.Linear(1024, 4096)
    
    # 2. Wrap it with our custom LoRA adapter
    lora_layer = LoRALinear(base_layer=base, r=8, lora_alpha=16)
    
    # 3. Pass dummy data through
    x = torch.randn((32, 1024))
    y = lora_layer(x)
    
    # Verify outputs
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify frozen/trainable parameters
    print(f"Base weight requires grad: {lora_layer.base_layer.weight.requires_grad}")
    print(f"LoRA A requires grad: {lora_layer.lora_A.requires_grad}")
    print(f"LoRA B requires grad: {lora_layer.lora_B.requires_grad}")
