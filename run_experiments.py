# %%
import time
import pandas as pd
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
import yaml
import clearml
from collections import deque

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

avg_time_per_experiment = 30  # minutes


# %%
PARAM_MAPPINGS = {
    # Basic parameters
    "config_name": ("--config-name", None),
    "model_name": ("model_name", "paths"),
    # Model parameters
    "d_model": ("d_model", "model"),
    "n_q_per_kv": ("n_q_per_kv", "model"),
    "n_kv": ("n_kv", "model"),
    "d_head": ("d_head", "model"),
    "d_ff": ("d_ff", "model"),
    "layers": ("layers", "model"),
    "vocab": ("vocab", "model"),
    "rope_max_timescale": ("rope_max_timescale", "model"),
    "a_attn": ("a_attn", "model"),
    "a_output": ("a_output", "model"),
    "zero_queries": ("zero_queries", "model"),
    "zero_unembed": ("zero_unembed", "model"),
    "parameterization": ("parameterization", "model"),
    "fully_aligned": ("fully_aligned", "model"),
    "gamma_embed": ("gamma_embed", "model"),
    "gamma_hidden": ("gamma_hidden", "model"),
    "gamma_unembed": ("gamma_unembed", "model"),
    # TODO: Missing base parameters
    "n_partitions": ("n_partitions", "model"),
    "apply_alibi": ("apply_alibi", "model"),
    "apply_rope": ("apply_rope", "model"),
    "hardmask_start_fraction": ("hardmask_start_fraction", "model"),
    "softmask_start_fraction": ("softmask_start_fraction", "model"),
    "hard_q_threshold": ("hard_q_threshold", "model"),
    "retrieve_budget": ("retrieve_budget", "model"),
    "initial_q_bias": ("initial_q_bias", "model"),
    # Training parameters
    "queue": ("queue", "training"),
    "warmup_steps": ("warmup_steps", "training"),
    "steps": ("steps", "training"),
    "steps_for_lr": ("steps_for_lr", "training"),
    "learning_rate": ("learning_rate", "training"),
    "batch_tokens": ("tokens.batch", "training"),
    "use_grad_clip": ("use_grad_clip", "training"),
    "use_gpu": ("use_gpu", "training"),
    "use_single_worker": ("use_single_worker", "training"),
    "n_log_iterations": ("n_log_iterations", "training"),
    "post_training_mult": ("post_training_mult", "training"),
}


# %%
def load_yaml_config(config_name):
    """
    Load the YAML config file for the given config name
    """
    config_path = Path("configs") / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_override_value(prefix, value, sub_param=None, param_in_config=False):
    """
    Format the override value based on its type and prefix
    """
    if pd.isna(value) or value == "":
        return None

    if isinstance(value, bool):
        value = str(value)
    elif isinstance(value, (int, float)):
        value = str(value)
    elif isinstance(value, str):
        if value.lower() in ["true", "false"]:
            value = value.capitalize()
        elif not prefix.startswith("--config-name"):  # Don't quote config name
            value = f'"{value}"'

    if sub_param:
        prefix_str = (
            f"{sub_param}.{prefix}" if param_in_config else f"+{sub_param}.{prefix}"
        )
        return f"{prefix_str}={value}"
    elif prefix.startswith("--"):
        return f"{prefix}={value}"
    return f"+{prefix}={value}"


def check_param_in_config(config, param_path):
    """
    Check if a parameter exists in the config
    param_path format: "section.param" e.g. "model.apply_alibi"
    """
    if not config:
        return False

    parts = param_path.split(".")
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    return True


def extract_clearml_id(output):
    """
    Extract ClearML task ID from subprocess output
    """
    match = re.search(r"ClearML Task: created new task id=([a-f0-9]+)", output)
    if match:
        return match.group(1)
    return None


def run_training(config_row, df, idx):
    """
    Run training command with the configuration from the CSV row
    """
    base_cmd = ["python", "-m", "train"]

    # Load the config file first
    config = load_yaml_config(config_row["config_name"])

    # Define parameter mappings (CSV column -> override parameter)

    # Build overrides list
    overrides = []
    for csv_param, (override_param, sub_param) in PARAM_MAPPINGS.items():
        if csv_param in config_row:
            # Check if parameter exists in config and needs ++ for override
            param_path = (
                f"{sub_param}.{override_param}" if sub_param else override_param
            )
            param_in_config = (
                check_param_in_config(config, param_path)
                if not override_param.startswith("--")
                else False
            )

            override = get_override_value(
                override_param, config_row[csv_param], sub_param, param_in_config
            )
            if override:
                if override.startswith("--"):
                    # Move config-name to front
                    overrides.insert(0, override)
                else:
                    overrides.append(override)

    cmd = base_cmd + overrides
    print("Running command:", " ".join(cmd))

    try:
        # Capture output of subprocess
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )

        # Extract ClearML ID from output
        clearml_id = extract_clearml_id(result.stdout)
        if clearml_id:
            print(f"Extracted ClearML Task ID: {clearml_id}")
            # Update the DataFrame with the ClearML ID
            df.at[idx, "task_id"] = clearml_id
            # Save the updated DataFrame
            df.to_csv("model_configs.csv", index=False)
        else:
            print("Warning: Could not find ClearML Task ID in output")

        # Print the output for logging purposes
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return clearml_id
    except subprocess.CalledProcessError as e:
        print(f"Error running training command: {e}")
        if e.output:
            print("Output:", e.output)
        if e.stderr:
            print("Error!", e.stderr)
        raise e


def get_config_value(config, param_path):
    """
    Get a value from the config dictionary using a dotted parameter path
    param_path format: "section.param" e.g. "model.apply_alibi"
    """
    if not config:
        return None

    parts = param_path.split(".")
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    return current


def update_with_config_values(df):
    """
    Update dataframe with values from config files where not explicitly overridden
    """
    # Create a copy of the dataframe
    updated_df = df.copy()

    # Iterate through each row
    for idx, row in df.iterrows():
        if pd.isna(row.get("config_name")):
            continue

        # Load the config for this row
        config = load_yaml_config(row["config_name"])
        if not config:
            continue

        # Check for each parameter if it's missing in the dataframe but exists in config
        for csv_param, (override_param, sub_param) in PARAM_MAPPINGS.items():
            # Skip config_name itself
            if csv_param == "config_name":
                continue

            # Skip if parameter is already set in dataframe
            if csv_param in df.columns and not pd.isna(row.get(csv_param)):
                continue

            # Form parameter path for config lookup
            param_path = (
                f"{sub_param}.{override_param}" if sub_param else override_param
            )

            # Get value from config
            if override_param.startswith("--"):
                # Handle special parameters that don't follow the same pattern
                continue

            config_value = get_config_value(config, param_path)

            # Add value to dataframe if found in config
            if config_value is not None:
                updated_df.at[idx, csv_param] = config_value

    return updated_df


# %%

# Read the CSV file
csv_path = Path("model_configs.csv")
if not csv_path.exists():
    print(f"Error: {csv_path} not found!")
    sys.exit(1)

# Read CSV with empty values as NaN
df = pd.read_csv(csv_path, na_values=["", "nan", "NaN"])
updated_df = update_with_config_values(df)
updated_df
# %%
# Filter out rows that already have a loss value (already run)
df_to_run = df[df["loss"].isna()]

print("Current loss values:")
print(df["loss"].to_string())

if df_to_run.empty:
    print("No new configurations to run!")

print(f"\nFound {len(df_to_run)} configurations to run")

experiments_to_run = deque([])
for idx, row in df_to_run.iterrows():
    print(f"\nRunning configuration {idx + 1}/{len(df_to_run)}")
    clearml_id = run_training(row, df, idx)
    if clearml_id:
        experiments_to_run.append(clearml_id)

print(f"Queued {len(experiments_to_run)} experiments")

# %%
# TODO: After we've queued the trainings listen to the ClearML queue till empty
# We will want to update the CSV with the final metrics when jobs complete
while experiments_to_run:
    print(f"Waiting for {len(experiments_to_run)} experiments to complete")
    time.sleep(avg_time_per_experiment * 60)
    print("Checking queue...")
    experiment: clearml.Task = clearml.Task.get_task(task_id=experiments_to_run[0])
    if experiment.status in ["completed", "failed"]:
        print(f"Experiment {experiments_to_run[0]} completed")
        experiments_to_run.popleft()
