from clearml import Task
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import yaml


TRACKED_FIELDS = [
    ("block_size", "Block_Size"),
    ("layers", "Concept_Decoder_Layers"),
    ("n_e_layers", "Encoder_Layers"),
    ("n_t_layers", "Token_Decoder_Layers"),
    ("reduction_strategy", "Reduction_Strategy"),
    ("n_kv", "Attention_Heads"),
    ("learning_rate", "Learning_Rate"),
    ("d_model", "d_model"),
    ("d_ff", "d_ff"),
]


def get_experiment_ids_from_url(url: str) -> list[str]:
    """Extract experiment IDs from a ClearML compare-experiments URL.

    Args:
        url (str): ClearML URL containing experiment IDs

    Returns:
        list[str]: List of experiment IDs
    """
    # Find the ids= parameter and extract everything after it until the next / or end of string
    if "ids=" not in url:
        raise ValueError("URL does not contain experiment IDs in the expected format")

    ids_section = url.split("ids=")[1].split("/")[0]
    experiment_ids = ids_section.split(",")
    return experiment_ids


def get_metrics_data(task_ids):
    metrics_data = {}
    config_data = {}
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        yaml_config = task.get_configuration_object("OmegaConf")
        config = yaml.safe_load(yaml_config)

        model_config = config["model"]
        training_config = config["training"]
        config_data[task_id] = {**model_config, **training_config}
        scalar_logs = task.get_reported_scalars()
        if "loss" not in scalar_logs:
            print(f"Loss not found for task {task_id} with name {task.name}")
            continue
        x_values = scalar_logs["loss"]["loss"]["x"]
        loss_values = scalar_logs["loss"]["loss"]["y"]

        final_loss = (
            None
            if "final_loss" not in scalar_logs
            else scalar_logs["final_loss"]["eval"]["y"]
        )
        final_perplexity = (
            None
            if "final_perplexity" not in scalar_logs
            else scalar_logs["final_perplexity"]["eval"]["y"]
        )

        task_name = task.name.replace("model.", "")
        metrics_data[task_id] = {
            "name": task_name,
            "steps": x_values,
            "loss": loss_values,
            "final_loss": final_loss,
            "final_perplexity": final_perplexity,
        }
    return metrics_data, config_data


def calculate_ema(data, smoothing=0.97):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(ema[-1] * smoothing + value * (1 - smoothing))
    return ema


def get_top_k_experiments(
    loss_data: dict, k: int = None, ema_smoothing: float = 0.97
) -> dict:
    """Get the top k experiments with lowest final EMA loss values.

    Args:
        loss_data (dict): Dictionary containing loss data for each experiment
        k (int, optional): Number of top experiments to return. If None, returns all.
        ema_smoothing (float, optional): Smoothing factor for EMA calculation. Defaults to 0.97.

    Returns:
        dict: Dictionary containing only the top k experiments
    """
    # Sort experiments by their final EMA loss value
    sorted_experiments = sorted(
        loss_data.items(),
        key=lambda x: calculate_ema(x[1]["loss"], smoothing=ema_smoothing)[
            -1
        ],  # Sort by final EMA value
    )

    # Take only top k if specified
    if k is not None:
        sorted_experiments = sorted_experiments[:k]

    return dict(sorted_experiments)


def plot_loss_data(
    loss_data,
    plot_last: int = 1000,
    ema_smoothing: float = 0.97,
    top_k: int = None,
    opacity: float = 0.25,
):
    # Get top k experiments if specified
    if top_k is not None:
        loss_data = get_top_k_experiments(
            loss_data, k=top_k, ema_smoothing=ema_smoothing
        )

    plt.figure(figsize=(10, 6))
    for _, data in loss_data.items():
        steps = data["steps"][-plot_last:]
        loss = data["loss"][-plot_last:]

        loss_ema = calculate_ema(loss, smoothing=ema_smoothing)

        # Plot EMA line first and use its color for the legend
        (ema_line,) = plt.plot(steps, loss_ema, label=f"{data['name']}")
        color = ema_line.get_color()

        # Plot raw loss with same color but lower alpha
        plt.plot(steps, loss, alpha=opacity, color=color)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(
        title="Experiments",
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        fontsize="small",
        title_fontsize="small",
        ncol=2,
    )
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()


def get_eval_metrics_table(metrics_data, config_data):
    """
    Filter valid data, parse model configuration from the `name` field using regex,
    and create a DataFrame sorted by final evaluation loss.

    Args:
        metrics_data (dict): Dictionary where each key is a GUID, and each value is a dict
                            containing model metadata, including 'name' and 'final_loss'.

    Returns:
        pd.DataFrame: DataFrame with parsed and formatted evaluation metrics.
    """

    # Parse the data
    data = []
    for guid, d in metrics_data.items():
        if d["final_loss"] is not None:
            name = d["name"]
            loss = float(d["final_loss"][0])

            # Match the name string with the regex pattern
            fields = {}
            config_fields = config_data[guid]
            # Convert numerical fields to appropriate types

            for config_key, display_key in TRACKED_FIELDS:
                if config_key in config_fields:
                    if config_key == "learning_rate":
                        fields[display_key] = float(config_fields[config_key])
                    elif config_key == "reduction_strategy":
                        fields[display_key] = config_fields[config_key]
                    else:
                        fields[display_key] = int(config_fields[config_key])
            data.append(
                {
                    "Name": name if len(name) < 25 else name[:25] + "...",
                    "Eval Loss": loss,
                    **fields,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by evaluation loss
    df = df.sort_values(by="Eval Loss", ascending=True).reset_index(drop=True)
    # Configure pandas display options for better visibility
    pd.set_option(
        "display.max_colwidth", 20
    )  # Limit column width to truncate long names
    pd.set_option("display.max_rows", None)  # Show all rows
    # pd.set_option("display.width", None)  # Auto-detect display width

    # Create formatters for different numeric types
    integer_columns = [
        "Block_Size",
        "Encoder_Layers",
        "Token_Decoder_Layers",
        "Concept_Decoder_Layers",
        "Attention_Heads",
        "Query_Per_Key_Value",
    ]
    float_columns = ["Learning_Rate", "Eval Loss"]
    string_columns = ["Name", "Reduction_Strategy"]

    formatters = {
        col: lambda x: f"{int(x)}" if pd.notnull(x) else "-" for col in integer_columns
    }
    formatters.update(
        {col: lambda x: f"{x:.6f}" if pd.notnull(x) else "-" for col in float_columns}
    )
    formatters.update(
        {col: lambda x: str(x) if pd.notnull(x) else "-" for col in string_columns}
    )

    # Create a styled DataFrame for better notebook display
    styled_df = (
        df.style.format(formatters)
        .set_properties(**{"text-align": "left", "white-space": "pre-wrap"})
        .set_properties(
            subset=["Name"],
            **{"max-width": "200px", "overflow": "hidden", "text-overflow": "ellipsis"},
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "left")]},
            ]
        )
    )

    return styled_df
