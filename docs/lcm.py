# %%
import plot_helper
import importlib

# %%
importlib.reload(plot_helper)
from plot_helper import *

# %%
TASK_IDS = [
    "58b7c60aa20e485d9c74a43819f720d5",
    "66b4c6f95b0e415aa26888bfc6efddf2",
    "ce8a074185ad4e91a9e9a721ff57f61a",
    "af54a412b286431fa2340e85c69ed293",
    "a7752a2b82254371a109e01c6661b271",
    "c6f92dce47d74c59a2a087aec3e34f41",
    "9c6b027deac04ce09822eecd02941420",
    "ca4c293546154ff9ad8da4446f22ddce",
    "e708aba99f9c4df7b21492e5b5366036",
    "cb467050886d4c0797d91ea6023ffd3d",
    "10e58923831e4f6e8efd18a8e00e83f8",
    "1d449329ac3f4d0fa3bf6d173f439c7b",
    "bb5b1ccedec745848754ca74ba60a28a",
    "0fd0a497fddf4223b7accdeecd5d6e35",
    "0bd01bebd9d340a29f691547dbf21426",
    "9d7ae33dd76e41f58e93a23ec123b7ca",
    "1c2d88bfe3a04e11829e693190c02001",
    "d3a41adb55954745b2a318ebc9032c1b",
    # "7a72052b8b13468c9275ba29b822bfb1",
    # "ad3801064a184e2994c25cf809c3dd8e",
]
TASK_IDS
# %%
metrics_data = get_metrics_data(TASK_IDS)
[(k, v["name"]) for (k, v) in metrics_data.items()]
# %%
plot_loss_data(metrics_data, plot_last=500, ema_smoothing=0.97, top_k=10)
plot_loss_data(metrics_data, plot_last=500, ema_smoothing=0.99, top_k=5)

# %%
top_k_metrics_data = get_top_k_experiments(metrics_data, k=10, ema_smoothing=0.97)
print("\n".join([f"{v['name']}" for (k, v) in top_k_metrics_data.items()]))

# %%
get_eval_metrics_table(metrics_data)

# %%
