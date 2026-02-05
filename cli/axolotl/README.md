# Axolotl Example

This example demonstrates how to run [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) on Serverless GPU Compute using SGCLI.

Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures including Llama, Mistral, and more.

## Prerequisites

1. A Hugging Face account with access to Llama models
2. Your HF token stored in the secrets vault at the path specified in `workload.yaml`

## Files

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies (Axolotl, Flash Attention, DeepSpeed)
- `fft-8b-liger-fsdp.yaml` - Axolotl training configuration for Llama-3.1-8B with FSDP and Liger kernels

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

- **FSDP** (Fully Sharded Data Parallel) for distributed training
- **Liger kernels** for optimized attention and activations
- **MLflow** for experiment tracking
- **FineTome-100k** dataset with chat template formatting

## Customization

### Training Configuration

Modify `fft-8b-liger-fsdp.yaml` to customize:

- `base_model` - Change the base model to fine-tune
- `datasets` - Modify dataset sources and formatting
- `sequence_len` - Adjust context length
- `micro_batch_size` / `gradient_accumulation_steps` - Control effective batch size
- `learning_rate` - Adjust training hyperparameters

### Multi-Node Scaling

The current configuration uses 16 GPUs across 2 nodes. To adjust in `workload.yaml`:

```yaml
compute:
  gpus: 16  # Total GPUs
  gpu_type: h100

command: |-
  accelerate launch \
    --num_machines 2 \        # Number of nodes
    --num_processes $WORLD_SIZE \
    ...
```

## Resources

- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Documentation](https://axolotl-ai-cloud.github.io/axolotl/)
- [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU)
