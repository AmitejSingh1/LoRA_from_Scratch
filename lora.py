import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    A custom PyTorch module that implements Low-Rank Adaptation (LoRA) for a linear layer.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_action = self.dropout(x)
        lora_action = F.linear(lora_action, self.lora_A)
        lora_action = F.linear(lora_action, self.lora_B)
        return base_output + (lora_action * self.scaling)

    def merge_and_unload(self) -> nn.Linear:
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta_w
        for param in self.base_layer.parameters():
            param.requires_grad = True
        return self.base_layer


class LoRAConv2d(nn.Module):
    """
    LoRA adapter for PyTorch Conv2d Vision Layers (like U-Net and VGG).
    We use two convolutions (A and B) to form a low-rank bottleneck.
    """
    def __init__(
        self, 
        base_layer: nn.Conv2d, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        
        # Freeze base parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        
        # A matrix: applies base kernel but collapses channels to rank `r`
        self.lora_A = nn.Conv2d(
            in_channels, r, 
            kernel_size=base_layer.kernel_size, 
            stride=base_layer.stride, 
            padding=base_layer.padding, 
            dilation=base_layer.dilation, 
            groups=base_layer.groups, 
            bias=False
        )
        
        # B matrix: 1x1 convolution expanding rank `r` to `out_channels`
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, stride=1, bias=False)
        
        self.dropout = nn.Dropout2d(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        # Standard init for A, zero init for B (so starting state is uncorrupted)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original heavy calculation (frozen)
        base_output = self.base_layer(x)
        
        # Parallel LoRA calculation
        lora_action = self.dropout(x)
        lora_action = self.lora_A(lora_action)
        lora_action = self.lora_B(lora_action)
        
        return base_output + (lora_action * self.scaling)
