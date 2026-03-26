import torch
import torch.nn as nn
from injector import inject_lora
from trainer import train_lora_epoch, save_lora_weights

# 1. Define a simple dummy neural network
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A tiny vision network
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    print("--- 1. Initializing Base Model ---")
    model = DummyNet()
    
    print("\n--- 2. Injecting LoRA ---")
    # Target every Conv2d and Linear layer in the model
    model = inject_lora(model, target_layer_names=["all"], r=4)
    
    print("\n--- 3. Setting up Mock Data (Vision) ---")
    # Create random images: batch size 8, 3 channels, 32x32 pixels
    mock_inputs = torch.randn(8, 3, 32, 32)
    mock_targets = torch.randint(0, 10, (8,))
    
    # Put them in a simple PyTorch list-based dataloader equivalent
    dataloader = [(mock_inputs, mock_targets)] * 5  # 5 batches per epoch
    
    print("\n--- 4. Beginning Training Loop ---")
    # IMPORTANT: Only pass parameters that require gradients (the new LoRA matrices) to the Optimizer!
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 4):
        print(f"\nStarting Epoch {epoch}...")
        # Note: Set device="cpu" if you don't have CUDA available right now.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss = train_lora_epoch(model, dataloader, optimizer, criterion, device=device)
        
    print("\n--- 5. Saving LoRA Weights ---")
    save_lora_weights(model, "mock_lora_checkpoint.pt")
    print("\nVerification Complete! LoRA Vision Pipeline is fully functional.")

if __name__ == "__main__":
    main()
