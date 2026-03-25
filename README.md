# LoRA From Scratch

A custom, from-scratch implementation of Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT) in PyTorch. 

## Features
- Drop-in `LoRALinear` replacement for `nn.Linear` layers.
- Memory-efficient training by freezing base model weights.
- Zero-overhead inference via weight merging (`merge_and_unload`).

## Project Structure
- `lora.py`: Contains the core mathematical implementation of the LoRA adapter.

## Why Built From Scratch?
While libraries like HuggingFace `PEFT` are great, building LoRA from the ground up demonstrates a deep understanding of Matrix mathematics, PyTorch module manipulation, and ML infrastructure.

## Next Steps
- [ ] Implement the dynamic injection mechanism to parse through arbitrary architectures.
- [ ] Build the Mixed-Precision (AMP) training loop.
- [ ] Add custom checkpointing to only save adapter weights (saving 99% of storage).
