# Databricks notebook source
# MAGIC %pip install -U packaging setuptools wheel ninja mlflow>=3.6
# MAGIC %pip install --no-build-isolation axolotl[flash-attn,deepspeed]
# MAGIC %pip install --force-reinstall --no-cache-dir --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
# MAGIC %pip install threadpoolctl==3.1.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add helpful messages to the log

# COMMAND ----------

import os
import subprocess
import sys

def print_system_info():
    """Print system information"""
    print_section("# SYSTEM INFORMATION REPORT")
    print_environment_variables()
    print_installed_packages()
    print_nvidia_smi()
    print_section("# END OF REPORT")


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def print_environment_variables():
    """Print all environment variables"""
    print_section("ENVIRONMENT VARIABLES")
    
    for key, value in sorted(os.environ.items()):
        print(f"{key}={value}")


def print_installed_packages():
    """Print all installed Python packages"""
    print_section("INSTALLED PYTHON PACKAGES")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting package list: {e}")
        print(e.stderr)


def print_nvidia_smi():
    """Print nvidia-smi output"""
    print_section("NVIDIA-SMI OUTPUT")
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA drivers may not be installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        print(e.stderr)

# COMMAND ----------

from serverless_gpu.launcher import distributed
from serverless_gpu.compute import GPUType

@distributed(gpus=16, gpu_type=GPUType.H100, remote=True)
def launch_job(**kwargs):
    import os
    import time
    from axolotl.cli.train import do_cli as do_train

    # TODO
    os.environ['HF_TOKEN'] = "replace_with_your_hf_token%"
    print_system_info()

    # TODO
    # Replace with your own databricks username
    # Choose the training config you want to use
    do_train("/Workspace/Users/my_databricks_username/databricks-SGC-Axolotl-examples/training_config/axolotl-fft-qwen32b-sdpa-fsdp-v2.yaml")   

# COMMAND ----------

launch_job.distributed()

# COMMAND ----------

