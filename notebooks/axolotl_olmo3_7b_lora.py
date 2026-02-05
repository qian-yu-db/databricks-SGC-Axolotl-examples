# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tune Olmo3 7B model with Axolotl on using Serverless GPU API
# MAGIC
# MAGIC <!--Author: Qian Yu-->
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
# MAGIC ## Install the required dependencies

# COMMAND ----------

# MAGIC %pip install -U packaging setuptools wheel ninja
# MAGIC %pip install mlflow>=3.6
# MAGIC %pip install --no-build-isolation axolotl[flash-attn]==0.12.2
# MAGIC %pip install transformers==4.57.3
# MAGIC %pip uninstall -y awq autoawq
# MAGIC %pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@0ee9ee8"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Huggingface token

# COMMAND ----------

import os

HF_TOKEN = dbutils.secrets.get(scope="my_secret_scope", key="hf_token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Configuration
# MAGIC * Download the configuration file from https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/olmo3/olmo3-7b-qlora.yaml
# MAGIC * Log metrics to MLflow
# MAGIC * Set checkpoint to UC Volume path
# MAGIC * Choose SDPA instead of Flash attention
# MAGIC * Save config to /Volumes/main/sgc-nightly/training_configs/axolotl/olmo3-7b-qlora.yaml

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "qyu_test")
dbutils.widgets.text("volume", "models")
dbutils.widgets.text("model", "olmo7b")

UC_CATALOG = dbutils.widgets.get("catalog")
UC_SCHEMA = dbutils.widgets.get("schema")
UC_VOLUME = dbutils.widgets.get("volume")
UC_MODEL_NAME = dbutils.widgets.get("model")

print(f"UC_CATALOG: {UC_CATALOG}")
print(f"UC_SCHEMA: {UC_SCHEMA}")
print(f"UC_VOLUME: {UC_VOLUME}")
print(f"UC_MODEL_NAME: {UC_MODEL_NAME}")

OUTPUT_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{UC_MODEL_NAME}"
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

# COMMAND ----------

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault

# Config is based on with some changes to fit GPU types
# https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/examples/olmo3/olmo3-7b-qlora.yaml

# Axolotl provides full control and transparency over model and training configuration
config = DictDefault(
    base_model="allenai/Olmo-3-7B-Instruct-SFT",
    plugins=[
        "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
    ],
    load_in_8bit=False,
    load_in_4bit=True,
    datasets=[
        {
            "path": "fozziethebeat/alpaca_messages_2k_test",
            "type": "chat_template"
        }
    ],
    dataset_prepared_path="last_run_prepared",
    val_set_size=0.1,
    output_dir=OUTPUT_DIR,
    adapter="qlora",
    lora_model_dir=None,
    sequence_len=2048,
    sample_packing=True,
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_linear=True,
    lora_target_modules=[
        "gate_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    wandb_project=None,
    wandb_entity=None,
    wandb_watch=None,
    wandb_name=None,
    wandb_log_model=None,
    gradient_accumulation_steps=4,
    micro_batch_size=2,
    num_epochs=1,
    optimizer="adamw_bnb_8bit",
    lr_scheduler="cosine",
    learning_rate=0.0002,
    bf16="auto",
    tf32=False,
    gradient_checkpointing=True,
    resume_from_checkpoint=None,
    logging_steps=1,
    flash_attention=False,
    warmup_ratio=0.1,
    evals_per_epoch=1,
    saves_per_epoch=1,
    # Eval dataset is too small
    eval_sample_packing=False,
    # Write metrics to MLflow
    use_mlflow=True,
    mlflow_tracking_uri="databricks",
    mlflow_run_name="olmo3-7b-qlora-axolotl",
    hf_mlflow_log_artifacts=False,
    wandb_mode="disabled",
    attn_implementation="sdpa",
    sdpa_attention=True,
    save_first_step=True,
    device_map=None,
)

# COMMAND ----------

from axolotl.utils import set_pytorch_cuda_alloc_conf

set_pytorch_cuda_alloc_conf()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use serverless_gpu python API to distribute
# MAGIC Wrap the Axolotl do_cli call with the @distributed decorator

# COMMAND ----------

from serverless_gpu.launcher import distributed
from serverless_gpu.compute import GPUType

@distributed(gpus=8, gpu_type=GPUType.H100, remote=False)
def run_train(cfg: DictDefault):
    import os
    os.environ['HF_TOKEN'] = HF_TOKEN

    from axolotl.common.datasets import load_datasets

    # Load, parse and tokenize the datasets to be formatted with qwen3 chat template
    # Drop long samples from the dataset that overflow the max sequence length

    # validates the configuration
    cfg = load_cfg(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    #os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"
    from axolotl.train import train

    # just train the first 16 steps for demo.
    # This is sufficient to align the model as we've used packing to maximize the trainable samples per step.
    cfg.max_steps = 16
    model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

    import mlflow
    mlflow_run_id = None
    if mlflow.last_active_run() is not None:
        mlflow_run_id = mlflow.last_active_run().info.run_id
    
    return mlflow_run_id

# COMMAND ----------

result = run_train.distributed(config)

# COMMAND ----------

run_id = result[0]
print(run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model to MLflow
# MAGIC Need to connect the notebook to H100 to load model checkpoint, otherwise CUDA out-of-memory error will be thrown.

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import os, json

BASE_ID = "allenai/Olmo-3-7B-Instruct-SFT"
FIXED_BASE_DIR = "/tmp/fixed_olmo3_7b_instruct_sft"  # choose any writable path

os.makedirs(FIXED_BASE_DIR, exist_ok=True)

# 1) Load original base model + tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

# 2) Fix generation_config on the *base model*
gen_cfg = base_model.generation_config
gen_cfg.do_sample = True           # match the recommended decoding
gen_cfg.temperature = 0.6
gen_cfg.top_p = 0.95

base_model.generation_config = gen_cfg

# 3) Save fixed base checkpoint
base_model.save_pretrained(FIXED_BASE_DIR)
tokenizer.save_pretrained(FIXED_BASE_DIR)

# 4) Optional sanity check
with open(os.path.join(FIXED_BASE_DIR, "generation_config.json")) as f:
    print(json.load(f))


# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import mlflow

torch.cuda.empty_cache()

print("Loading fixed base + LoRA adapter for registration...")

FIXED_BASE_DIR = "/tmp/fixed_olmo3_7b_instruct_sft"

# Use fixed base instead of HF_MODEL_NAME
base_model = AutoModelForCausalLM.from_pretrained(
    FIXED_BASE_DIR,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(FIXED_BASE_DIR)

adapter_dir = OUTPUT_DIR  # your LoRA adapter path
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

# Merge LoRA into base and drop PEFT wrappers
merged_model = peft_model.merge_and_unload()

# Build pipeline from merged model
text_gen_pipe = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
)

# Ensure final generation_config is consistent (should already be)
gen_cfg = text_gen_pipe.model.generation_config
gen_cfg.do_sample = True
gen_cfg.temperature = 0.6
gen_cfg.top_p = 0.95
text_gen_pipe.model.generation_config = gen_cfg


# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
try:
    from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation
except ImportError:
    from transformers.activations import NewGELUActivation, GELUTanh as PytorchGELUTanh, GELUActivation

from peft import PeftModel
import mlflow
from mlflow import transformers as mlflow_transformers
import torch

torch.cuda.empty_cache()
# Load the trained model for registration
print("Loading LoRA model for registration...")
# For LoRA models, we need both base model and adapter

HF_MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
base_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    trust_remote_code=True
)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
adapter_dir = OUTPUT_DIR
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
# Merge LoRA into base and drop PEFT wrappers
merged_model = peft_model.merge_and_unload()

components = {
    "model": merged_model,
    "tokenizer": tokenizer,
}

# Create Unity Catalog model name
full_model_name = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

print(f"Registering model as: {full_model_name}")

text_gen_pipe = pipeline(
    task="text-generation",
    model=peft_model,
    tokenizer=tokenizer,
)

# COMMAND ----------

with mlflow.start_run(run_id=run_id):
    model_info = mlflow.transformers.log_model(
        transformers_model=text_gen_pipe,
        name="olmo3-7b-qlora",
        task="llm/v1/completions",
        registered_model_name=full_model_name
    )

# COMMAND ----------

