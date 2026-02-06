# TRL SFT Example

This example demonstrates how to run supervised fine-tuning (SFT) using [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) on Serverless GPU Compute with SGCLI.

TRL is a full-stack library providing tools for training transformer language models with Reinforcement Learning, from supervised fine-tuning to RLHF and beyond.

## Prerequisites

1. A Hugging Face account with access to Llama models
2. Your HF token stored in the secrets vault at the path specified in `workload.yaml`

## Files

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies (TRL, PEFT, Flash Attention, Liger Kernel)

## Usage

Run the example with:

```bash
sgcli run -f workload.yaml
```

Or watch the logs in real-time:

```bash
sgcli run -f workload.yaml --watch
```

## Configuration Details

This example fine-tunes **Meta-Llama-3.1-8B** using:

- **TRL SFT Trainer** for supervised fine-tuning
- **LoRA/PEFT** for parameter-efficient fine-tuning
- **FSDP** (Fully Sharded Data Parallel) for distributed training
- **Flash Attention 2** for optimized attention computation
- **Liger Kernel** for optimized training operations
- **MLflow** for experiment tracking
- **UltraChat 200k** dataset for instruction tuning

## Customization

### Training Configuration

#### Model Settings
```yaml
model_name_or_path: meta-llama/Llama-3.1-8B  # Change base model
attn_implementation: flash_attention_2        # Attention implementation
torch_dtype: bfloat16                         # Data type
```

#### Dataset Settings
```yaml
dataset_name: HuggingFaceH4/ultrachat_200k   # Dataset from HF Hub
dataset_train_split: train_sft                # Training split
dataset_test_split: test_sft                  # Evaluation split
```

#### Training Hyperparameters
```yaml
num_train_epochs: 1                           # Number of epochs
per_device_train_batch_size: 2                # Batch size per GPU
gradient_accumulation_steps: 4                # Gradient accumulation
learning_rate: 2.0e-5                         # Learning rate
max_seq_length: 2048                          # Max sequence length
```

#### LoRA Configuration
```yaml
use_peft: true                                # Enable LoRA
lora_r: 16                                    # LoRA rank
lora_alpha: 32                                # LoRA alpha
lora_dropout: 0.05                            # LoRA dropout
lora_target_modules:                          # Modules to apply LoRA
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

### Full Fine-Tuning

To perform full fine-tuning instead of LoRA, set in `config.yaml`:

```yaml
use_peft: false
```

### Multi-Node Scaling

The current configuration uses 8 H100 GPUs on 1 node. To scale to multiple nodes, update `workload.yaml`:

```yaml
compute:
  gpus: 16  # Total GPUs across all nodes
  gpu_type: h100

command: |-
  accelerate launch \
    --num_machines 2 \              # Number of nodes
    --num_processes $WORLD_SIZE \
    ...
```

### Using Custom Datasets

TRL supports multiple dataset formats. You can use datasets from Hugging Face Hub or local datasets:

```yaml
# Using a different HF dataset
dataset_name: your-org/your-dataset
dataset_train_split: train
dataset_test_split: validation
```

For custom data formats, you can also use the `datasets` field with mixture configurations. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/sft_trainer) for more details.

## Advanced Features

### Gradient Checkpointing

Enabled by default to reduce memory usage:

```yaml
gradient_checkpointing: true
```

### Mixed Precision Training

Using bfloat16 for better numerical stability:

```yaml
bf16: true
torch_dtype: bfloat16
```

### Model Pushing to Hub

To push your trained model to the Hugging Face Hub:

```yaml
push_to_hub: true
hub_model_id: your-username/model-name
```

## Resources

- [TRL GitHub](https://github.com/huggingface/trl)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL SFT Trainer Guide](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU)

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
- Reducing `per_device_train_batch_size`
- Increasing `gradient_accumulation_steps`
- Enabling `gradient_checkpointing: true`
- Reducing `max_seq_length`

### Flash Attention Issues

If Flash Attention installation fails, you can fall back to standard attention:

```yaml
attn_implementation: eager
```

Then remove `flash-attn` from `requirements.yaml`.
