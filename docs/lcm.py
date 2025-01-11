# %%
from plot_helper import *

# %%
TASK_IDS = {
    "1d449329ac3f4d0fa3bf6d173f439c7b": "slim_v4-32_31m_block_size=2_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_layers=5",
    "1e2d44c3bbad48c3a805d94478902f1c": "slim_v4-32_31m_lcm_lr:0.020380",
    "cb467050886d4c0797d91ea6023ffd3d": "slim_v4-32_31m_block_size=1_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_base.n_q_per_kv=1_layers=5_n_q_per_kv=1",
    "10e58923831e4f6e8efd18a8e00e83f8": "slim_v4-32_31m_block_size=1_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_base.n_q_per_kv=1_layers=5_n_q_per_kv=1",
    "e708aba99f9c4df7b21492e5b5366036": "slim_v4-32_31m_block_size=1_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_base.n_q_per_kv=1_layers=5_n_q_per_kv=1",
    "ca4c293546154ff9ad8da4446f22ddce": "slim_v4-32_31m_block_size=2_n_e_layers=1_n_t_layers=3_reduction_strategy=attn_base.n_q_per_kv=2_layers=4_n_q_per_kv=2",
    "9c6b027deac04ce09822eecd02941420": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=4_reduction_strategy=attn_base.n_q_per_kv=4_layers=5_n_q_per_kv=4",
    "c6f92dce47d74c59a2a087aec3e34f41": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=1_reduction_strategy=attn_base.n_q_per_kv=2_layers=6_n_q_per_kv=2",
    "a7752a2b82254371a109e01c6661b271": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_base.n_q_per_kv=2_layers=5_n_q_per_kv=2",
    "af54a412b286431fa2340e85c69ed293": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=3_reduction_strategy=attn_base.n_q_per_kv=4_layers=4_n_q_per_kv=4",
    "ce8a074185ad4e91a9e9a721ff57f61a": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=1_reduction_strategy=attn_base.n_q_per_kv=4_layers=6_n_q_per_kv=4",
    "66b4c6f95b0e415aa26888bfc6efddf2": "slim_v4-32_31m_block_size=2_n_e_layers=1_n_t_layers=1_reduction_strategy=attn_base.d_ff=2048_base.n_q_per_kv=2_d_ff=2048_layers=6_n_q_per_kv=2",
    "58b7c60aa20e485d9c74a43819f720d5": "slim_v4-32_31m_block_size=4_n_e_layers=1_n_t_layers=2_reduction_strategy=attn_base.n_q_per_kv=4_layers=5_n_q_per_kv=4",
    # TODO: add baseline
}

# %%
loss_data = get_loss_data(TASK_IDS.keys())
plot_loss_data(loss_data, plot_last=1000)

# %%
