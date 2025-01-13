# %%
from plot_helper import *

# %%
experiment_ids = [
    "483668aecfb34f0e98ab13a69270c31f",
    "d4122017e5814f0d94102b8db258b149",
    "8607ffb881a9418cb8276e820e7cf8c5",
    "cb20112968a44b9387f9a5e404f22942",
]
# %%
loss_data = get_loss_data(experiment_ids)
[(k, v["name"]) for (k, v) in loss_data.items()]
# %%
plot_loss_data(loss_data, plot_last=125)

# %% [markdown]
"""
We essentially see that the number of clusters doesn't seem to worsen loss and slightly improves it.
"""

# %%
