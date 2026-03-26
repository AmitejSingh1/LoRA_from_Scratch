# LoRA From Scratch (PyTorch)

A custom, from-scratch implementation of Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT), built strictly in PyTorch. 

## Novel Features
- **Computer Vision Support (`LoRAConv2d`)**: Unlike standard LoRA tutorials that only apply to Transformers, this framework natively supports `nn.Conv2d`. You can easily fine-tune ResNets, U-Nets, and large medical imaging models directly.
- **Dynamic Layer Injection**: Recursively parses arbitrary model architectures and surgically swaps target layers (`nn.Linear` or `nn.Conv2d`) with LoRA variants.
- **Memory-Efficient Training Loop**: Includes a `torch.amp` (Mixed Precision) pipeline built for 8GB consumer GPUs (like the RTX 4060).
- **Custom Checkpointing**: Extracts and saves *only* the low-rank matrices, condensing model checkpoints from Gigabytes to Megabytes.

## Project Structure
- `lora.py`: Contains both `LoRALinear` and `LoRAConv2d` mathematical implementations.
- `injector.py`: The surgery script to dynamically inject LoRA into base models.
- `trainer.py`: The mixed precision training and state-dict extraction logic.
- `main.py`: End-to-end verification script proving parameter savings and loss reduction.

## Quick Start
Run the verification script to see the parameter savings and training loop in action:
```bash
python main.py
```
