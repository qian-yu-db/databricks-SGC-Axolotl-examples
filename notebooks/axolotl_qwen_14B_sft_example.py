# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tune Qwen3 14B with Axolotl
# MAGIC
# MAGIC [<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
# MAGIC
# MAGIC Axolotl is the most performant LLM post-training framework available, delivering faster training with efficient, consistent and stable performance. Train your workload and ship your product 30% faster; saving you both time and money.
# MAGIC
# MAGIC - ⭐ us on [GitHub](https://github.com/axolotl-ai-cloud/axolotl)
# MAGIC - 📜 Read the [Docs](http://docs.axolotl.ai/)
# MAGIC - 💬 Chat with us on [Discord](https://discord.gg/mnpEYgRUmD)
# MAGIC - 📰 Get updates on [X/Twitter](https://x.com/axolotl_ai)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Installation
# MAGIC
# MAGIC Axolotl is easy to install from [pip](https://pypi.org/project/axolotl/), or use our [pre-built Docker images](http://docs.axolotl.ai/docs/docker.html) for a hassle free dependency experience. See our [docs](http://docs.axolotl.ai/docs/installation.html) for more information.

# COMMAND ----------

# This step can take ~5-10 minutes to install dependencies
%pip install --no-build-isolation axolotl[flash-attn]>=0.9.1
%pip install -U "xformers==0.0.31"
%pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f643b88"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo: Talk Like a Pirate
# MAGIC
# MAGIC In this demo, we are training the model ***to respond like a pirate***. This was chosen as a way to easily show how to train a model to respond in a certain style of your choosing (without being prompted) and is quite easy to validate within the scope of a Colab.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload your own dataset or use a Huggingface dataset
# MAGIC
# MAGIC You can choose to use your own JSONL file from your own [Google Drive](https://drive.google.com/drive/home); for example downloading the [Pirate-Ultrachat JSONL](https://huggingface.co/datasets/winglian/pirate-ultrachat-10k/blob/main/train.jsonl) to your Google Drive. JSONL datasets should be formatted similar to the [OpenAI dataset format](https://cookbook.openai.com/examples/chat_finetuning_data_prep).
# MAGIC
# MAGIC You can also simply use the [`winglian/pirate-ultrachat-10k`](https://huggingface.co/datasets/winglian/pirate-ultrachat-10k) dataset directly.
# MAGIC

# COMMAND ----------

# Default to HF dataset location
dataset_id = "winglian/pirate-ultrachat-10k"
uploaded = {}

# COMMAND ----------

# MAGIC %md
# MAGIC # Configure for Supervised Fine-Tuning (SFT)

# COMMAND ----------

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault

# Axolotl provides full control and transparency over model and training configuration
config = DictDefault(
    base_model="Qwen/Qwen3-14B",  # Use the instruct tuned model, but we're aligning it to be a pirate
    load_in_4bit=True,  # set to True for qLoRA
    adapter="qlora",
    lora_r=32,
    lora_alpha=64,
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # train self_attn linear modules
        "gate_proj",
        "down_proj",
        "up_proj",  # train MLP linear modules
    ],
    lora_qkv_kernel=True,  # optimized triton kernels for LoRA
    lora_o_kernel=True,
    lora_mlp_kernel=True,
    embeddings_skip_upcast=True,  # keep embeddings in fp16 so the model fits in 15GB VRAM
    xformers_attention=False,  # use xformers on Colab w/ T4 for memory efficient attention, flash_attention only on Ampere or above
    plugins=[
        # more efficient training using Apple's Cut Cross Entropy; https://github.com/apple/ml-cross-entropy
        "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
    ],
    sample_packing=True,  # 2-6x increase in tokens per micro-batch
    # when using packing, use a slightly higher learning rate to account for fewer steps
    # alternatively, reduce the micro_batch_size + gradient_accumulation_steps to achieve closer to the same number of steps/epoch
    learning_rate=0.00019,
    sequence_len=4096,  # larger sequence length improves packing efficiency for more tokens/sec
    micro_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # tradeoff reduced VRAM for increased time
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },
    optimizer="paged_adamw_8bit",
    lr_scheduler="cosine",
    warmup_steps=5,
    fp16=True,  # use float16 + automatic mixed precision, bfloat16 not supported on Colab w/ T4
    bf16=False,
    max_grad_norm=0.1,  # gradient clipping
    num_epochs=1,
    saves_per_epoch=2,  # how many checkpoints to save over one epoch
    logging_steps=1,
    output_dir="./outputs/qwen-sft-pirate-rrr",
    chat_template="qwen3",
    datasets=[
        {
            "path": dataset_id,  # Huggingface Dataset id or path to train.jsonl
            "type": "chat_template",
            "split": "train",
            "eot_tokens": ["<|im_end|>"],
        }
    ],
    dataloader_prefetch_factor=8,  # dataloader optimizations
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    use_mlflow=True,
    mlflow_tracking_uri="databricks",
    mlflow_run_name="qwen3-14b-axolotl",
    hf_mlflow_log_artifacts=False,
    attn_implementation="sdpa",
    sdpa_attention=True,
)

# validates the configuration
cfg = load_cfg(config)

# COMMAND ----------

from axolotl.utils import set_pytorch_cuda_alloc_conf

set_pytorch_cuda_alloc_conf()

# COMMAND ----------

# MAGIC %md
# MAGIC # Datasets
# MAGIC
# MAGIC Axolotl has a robust suite of loaders and transforms to parse most open datasets of any format into the appropriate chat template for your model. Axolotl will mask input tokens from the user's prompt so that the train loss is only calculated against the model's response. For more information, [see our documentation](http://docs.axolotl.ai/docs/dataset-formats/conversation.html) on dataset preparation.
# MAGIC

# COMMAND ----------

from axolotl.common.datasets import load_datasets

# Load, parse and tokenize the datasets to be formatted with qwen3 chat template
# Drop long samples from the dataset that overflow the max sequence length
dataset_meta = load_datasets(cfg=cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training
# MAGIC
# MAGIC

# COMMAND ----------

from axolotl.train import train
import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Workspace/Users/my_databricks_user_name/mlflow_experiments/qwen-sft-pirate") # TODO replace with your own experiment path

# just train the first 25 steps for demo.
# This is sufficient to align the model as we've used packing to maximize the trainable samples per step.
cfg.max_steps = 25

with mlflow.start_run(
    run_name='qwen_sft_pirate',
    log_system_metrics=True
):
  model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

# COMMAND ----------

# MAGIC %md
# MAGIC # Inferencing the trained model

# COMMAND ----------

from transformers import TextStreamer

messages = [
    {
        "role": "user",
        "content": "Explain the Pythagorean theorem to me.",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)

outputs = model.generate(
    **tokenizer(prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=192,
    temperature=1.0,
    top_p=0.8,
    top_k=32,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print(outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

mlflow_run_id = mlflow.last_active_run().info.run_id

full_model_name = "main.my_schema.qwen-sft-pirate" # TODO replace with your own UC model path

try:
    with mlflow.start_run(run_id = mlflow_run_id):
        model_info = mlflow.transformers.log_model(
            transformers_model={'model': model, 'tokenizer': tokenizer},
            name='model',
            registered_model_name=full_model_name,
            await_registration_for=3600,
            task='llm/v1/chat',
        )
    print(f"✓ Model successfully registered in Unity Catalog: {full_model_name}")
    print(f"✓ MLflow model URI: {model_info.model_uri}")
    print(f"✓ Model version: {model_info.version}")
except Exception as e:
    print(f"✗ Model registration failed: {e}")
    print("Model is still saved locally and can be registered manually")
    print(f"Local model path: {cfg['output_dir']}")
