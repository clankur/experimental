"""Main training loop, including the model, loss function, and optimizer."""

from collections import namedtuple
import operator
import os
import signal
import subprocess
import time

import env

env.set_variables()
import shardlib.shardtypes as shardtypes

shardtypes.register_with_typeguard()
import gcsfs  # Needed for clearml setup

import datetime
from functools import cached_property, partial
from typing import Any, Optional, Tuple, Union
import hydra
from typeguard import typechecked
from dataclasses import dataclass, replace
import jax
from jax import lax
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import math
from input_loader import (
    FlatTokensParams,
    HuggingFaceDataParams,
    TokenBatch,
    TokenBatchParams,
    get_loader,
)
from shardlib.shardtypes import (
    bf16,
    bool_,
    f32,
    pytree_dataclass,
    u32,
    make_shardings,
    Array,
)
import shardlib.shardops as shardops

P = PartitionSpec
import einops
import jax_extra
from jax_extra import fold_in_str, explicit_activation_checkpointing, save_for_backward
import os
import training_io
from clearml import Task
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.tree_util import tree_leaves

PRNGKey = Any


# TODO:
# setup cluster centroids
# compute q_alignment = q dot centroid,  k_alignment = k dot centroid
# reduce q_alignment, k_alignment to scalar values
# loss -= q_alignment + k_alignment
# lax.stop_gradient(q_alignment)
# lax.stop_gradient(k_alignment)


@dataclass(frozen=True)
class BaseWidths:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    d_ff: int


@dataclass(frozen=True)
class Hparams:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    d_ff: int

    vocab: int
    layers: int
    base: BaseWidths
    n_clusters: int

    # parameters for mup
    a_attn: float
    a_output: float
    zero_queries: bool
    zero_unembed: bool

    # parameters for exp scaling
    parameterization: str
    fully_aligned: bool
    gamma_embed: float
    gamma_hidden: float
    gamma_unembed: float

    # fields for position embeddings
    rope_max_timescale: int
    apply_rope: Optional[bool] = False
    apply_alibi: Optional[bool] = False

    clustering_start_fraction: float = 0.0


def get_parameterization(style: str, fully_aligned: bool = True):
    Parameterization = namedtuple(
        "Parameterization",
        [
            "embed_init_var",
            "embed_param_mult",
            "embed_lr",
            "embed_grad",
            "hidden_init_var",
            "hidden_param_mult",
            "hidden_lr",
            "hidden_grad",
            "unembed_init_var",
            "unembed_param_mult",
            "unembed_lr",
            "unembed_grad",
        ],
    )

    base_params = {
        "sp": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=0.5,
            unembed_init_var=1.0,
            unembed_param_mult=0.0,
            unembed_lr=1.0,
            unembed_grad=0.0,
        ),
        "mup": Parameterization(
            embed_init_var=1.0,
            embed_param_mult=0.5,
            embed_lr=0.5,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=1.0,
            unembed_init_var=1.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "ntk": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.0,
            unembed_init_var=0.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "mean-field": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=1.0,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.5,
            unembed_init_var=0.0,
            unembed_param_mult=1.0,
            unembed_lr=0.0,
            unembed_grad=1.0,
        ),
    }

    style = style.lower()
    if style not in base_params:
        raise ValueError(f"Unknown parameterization style: {style}")

    params = base_params[style]._asdict()

    if not fully_aligned:
        if style == "sp":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.5,
                }
            )
        elif style == "mup":
            params.update(
                {
                    "embed_lr": 0.5,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "ntk":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "mean-field":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": -0.5,
                }
            )

    return Parameterization(**params)


@pytree_dataclass
class SyntheticMetrics:
    avg_confidence: f32[b""]
    avg_char_confidence: f32[b""]
    max_char_confidence: f32[b""]
    avg_start_char_confidence: f32[b""]
    avg_final_char_confidence: f32[b""]


@pytree_dataclass
class TransformerLayer:
    ln1: f32["d_model/t/d"]
    ln2: f32["d_model/t/d"]
    ln_q_nope: f32["d_head/t/d"]
    ln_k_nope: f32["d_head/t/d"]
    w_q: f32["d_model/d n_q_per_kv n_kv/t d_head"]
    w_kv: f32["2 d_model/d n_kv/t d_head"]
    w_o: f32["d_model/d n_q_per_kv n_kv/t d_head"]
    w_gate: f32["d_model/d d_ff/t"]
    w_up: f32["d_model/d d_ff/t"]
    w_down: f32["d_model/d d_ff/t"]
    clusters: f32["n_clusters n_q_per_kv n_kv/t d_head/d"]


Transformer = Array["layers", TransformerLayer]


@pytree_dataclass
class Model:
    embed: f32["vocab/t d_model/d"]
    unembed: f32["vocab/t d_model/d"]
    transformer: Transformer
    final_layer_norm: f32["d_model/d/t"]

    @staticmethod
    @typechecked
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        # https://github.com/google/jax/issues/20390 for ones_like with sharding.
        ln1 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln2 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_q_nope = jnp.ones((h.layers, h.d_head), dtype=jnp.float32)
        ln_k_nope = jnp.ones((h.layers, h.d_head), dtype=jnp.float32)
        final_layer_norm = jnp.ones((h.d_model,), dtype=jnp.float32)

        # All of wi/wq/wo/wo/w_kv use truncated_normal initializers with 'fan_in' scaling,
        # i.e. variance set to 1.0/fan_in.
        # The constant is stddev of standard normal truncated to (-2, 2)
        truncated_normal_stddev = 0.87962566103423978
        p = get_parameterization(h.parameterization)
        base = h.base

        embed_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.embed_init_var)
        # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
        d_model_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.hidden_init_var)

        target_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        base_head_dim = base.n_q_per_kv * base.n_kv * base.d_head
        w_o_scale = (
            math.sqrt(base_head_dim) / (target_head_dim * truncated_normal_stddev)
        ) ** (p.hidden_init_var)
        w_up_scale = d_model_scale
        w_down_scale = (math.sqrt(base.d_ff) / (h.d_ff * truncated_normal_stddev)) ** (
            p.hidden_init_var
        )

        ff_shape = (h.layers, h.d_model, h.d_ff)
        embed = embed_scale * jax.random.normal(
            jax_extra.fold_in_str(rng, "embed"), (h.vocab, h.d_model), dtype=jnp.float32
        )
        w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_gate"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_up = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_up"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_down"), -2, 2, ff_shape, dtype=jnp.float32
        )

        w_q_scale = d_model_scale
        w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
        w_o_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)

        if h.zero_queries:
            w_q = jnp.zeros(w_q_shape, dtype=jnp.float32)
        else:
            w_q = w_q_scale * jax.random.truncated_normal(
                fold_in_str(rng, "w_q"), -2, 2, w_q_shape, dtype=jnp.float32
            )

        w_kv = w_q_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_kv"), -2, 2, w_kv_shape, dtype=jnp.float32
        )

        w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_o"), -2, 2, w_o_shape, dtype=jnp.float32
        )

        unembed_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.unembed_init_var)

        if h.zero_unembed:
            unembed = jnp.zeros((h.vocab, h.d_model), dtype=jnp.float32)
        else:
            unembed = unembed_scale * jax.random.truncated_normal(
                fold_in_str(rng, "unembed"),
                -2,
                2,
                (h.vocab, h.d_model),
                dtype=jnp.float32,
            )

        clusters_shape = (h.layers, h.n_clusters, h.n_q_per_kv, h.n_kv, h.d_head)
        clusters = d_model_scale * jax.random.truncated_normal(
            fold_in_str(rng, "clusters"), -2, 2, clusters_shape, dtype=jnp.float32
        )
        # Normalize clusters to have L2 norm of 1 along the half_D dimension
        clusters = clusters / jnp.linalg.norm(clusters, axis=-1, keepdims=True)

        arrays = Model(
            embed=embed,
            unembed=unembed,
            transformer=Transformer(
                ln1=ln1,
                ln2=ln2,
                ln_q_nope=ln_q_nope,
                ln_k_nope=ln_k_nope,
                w_q=w_q,
                w_kv=w_kv,
                w_o=w_o,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                clusters=clusters,
            ),
            final_layer_norm=final_layer_norm,
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    @typechecked
    def forward_pass(
        self,
        h: Hparams,
        ids: u32[b"B/d L"],
        is_seq_start: bool_[b"B/d L"],
        use_clustering: u32[b""],
    ) -> Tuple[f32[b"B/d L V/t"], Tuple[f32[b""], f32[b""], f32[b""]]]:
        p = get_parameterization(h.parameterization)
        embed_mult = (h.d_model / h.base.d_model) ** -p.embed_param_mult
        hidden_mult = (h.d_model / h.base.d_model) ** -p.hidden_param_mult
        unembed_mult = (h.d_model / h.base.d_model) ** -p.unembed_param_mult

        # Initial embedding lookup.
        embed = embed_mult * shardops.all_gather(
            "V/t M/d -> V/t M", jnp.bfloat16(self.embed)
        )
        one_hot_ids = jax.nn.one_hot(ids, self.embed.shape[0])
        x = shardops.einsum_unreduced("B/d L V/t, V/t M -> B/d L M", one_hot_ids, embed)
        x = shardops.psum_scatter("B/d L M -> B/d L M/t", x)

        L = ids.shape[1]
        segment_ids = jnp.cumsum(is_seq_start, axis=1)
        segment_mask: bool_[b"B/d L L"] = (
            segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
        )
        segment_mask: bool_[b"B/d L L 1 1"] = segment_mask[
            ..., jnp.newaxis, jnp.newaxis
        ]  # add axes for q_per_k, num_kv_heads dimensions
        causal_mask: bool_[b"1 L L 1 1"] = jnp.tril(
            jnp.ones((L, L), dtype=jnp.bool_), 0
        )[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        causal_mask: bool_[b"B/d L L 1 1"] = jnp.logical_and(segment_mask, causal_mask)

        if h.apply_rope:
            rope_table = RopeTable.create(L, h)

        if h.apply_alibi:
            alibi = Alibi.create(h)

        ##### Transformer blocks.
        @explicit_activation_checkpointing
        # @typechecked
        def loop_body(
            x: bf16[b"B/d L M/t"], layer_weights: TransformerLayer
        ) -> Tuple[bf16[b"B/d L M/t"], f32[b""], Tuple[()]]:
            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln1))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            w_q = shardops.all_gather(
                "M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_q)
            )
            q = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d L M, M Q K/t D -> B/d L Q K/t D", nx, w_q
                )
            )
            if h.apply_rope:
                q = rope_table.apply("L D -> 1 L 1 1 D", q)

            w_kv = shardops.all_gather(
                "2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(layer_weights.w_kv)
            )
            k, v = hidden_mult * shardops.einsum_unreduced(
                "B/d L M, k_v M K/t D -> k_v B/d L K/t D", nx, w_kv
            )
            k = save_for_backward(k)
            v = save_for_backward(v)
            if h.apply_rope:
                k = rope_table.apply("L d -> 1 L 1 d", k)

            logit_scale = jax.lax.select(
                h.parameterization.lower() == "mup",
                h.a_attn * math.sqrt(h.base.d_head) / h.d_head,
                1.0 / math.sqrt(h.d_head),
            )
            logits = logit_scale * shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, B/d Klen K/t D -> B/d Qlen Klen Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )
            # clustering
            clusters = shardops.all_gather(
                "n_clusters Q K/t D/d -> n_clusters Q K/t D",
                layer_weights.clusters,
            )
            # Normalize clusters to have L2 norm of 1 along the D dimension
            clusters = clusters / jnp.linalg.norm(clusters, axis=-1, keepdims=True)

            cluster_alignment = shardops.einsum_unreduced(
                "n_clusters Q K/t D, n_clusters2 Q K/t D -> Q K/t n_clusters n_clusters2",
                clusters,
                clusters,
            )
            # Mask out self-alignment (diagonal) since it will always be 1
            cluster_mask = 1.0 - jnp.eye(h.n_clusters)[None, None]
            cluster_alignment = cluster_alignment * cluster_mask
            # Average over non-diagonal elements
            # Special case for n_clusters = 1 to avoid division by zero
            if h.n_clusters > 1:
                avg_cluster_alignment = einops.reduce(
                    cluster_alignment,
                    "Q K n_clusters n_clusters2 -> Q K n_clusters",
                    "sum",
                ) / (h.n_clusters - 1)
            else:
                # For n_clusters = 1, there are no non-diagonal elements, so set to 0
                avg_cluster_alignment = jnp.zeros_like(cluster_alignment)

            # Apply layer norm to q and k before clustering
            ln_q_nope = shardops.all_gather(
                "D/t/d -> D", jnp.float32(layer_weights.ln_q_nope)
            )
            ln_k_nope = shardops.all_gather(
                "D/t/d -> D", jnp.float32(layer_weights.ln_k_nope)
            )
            nq = jnp.bfloat16(rms_norm(q) * ln_q_nope)  # B/d L Q K/t D
            nk = jnp.bfloat16(rms_norm(k) * ln_k_nope)  # B/d L K/t D

            # Compute alignments with clusters
            q_alignment = shardops.einsum_unreduced(
                "B/d L Q K/t D, n_clusters Q K/t D -> B/d Q K/t L n_clusters",
                lax.stop_gradient(nq),
                clusters,
            )
            k_alignment = shardops.einsum_unreduced(
                "B/d L K/t D, n_clusters Q K/t D -> B/d Q K/t L n_clusters",
                lax.stop_gradient(nk),
                clusters,
            )
            q_to_cluster = jnp.argmax(q_alignment, axis=-1)
            k_to_cluster = jnp.argmax(k_alignment, axis=-1)
            cluster_q_alignment_mask = einops.rearrange(
                jax.nn.one_hot(q_to_cluster, h.n_clusters),
                "B Q K Qlen n_clusters ->  B Q K n_clusters Qlen 1",  # Broadcast to match attention dims
            )
            cluster_k_alignment_mask = einops.rearrange(
                jax.nn.one_hot(k_to_cluster, h.n_clusters),
                "B Q K Klen n_clusters -> B Q K n_clusters Klen 1",  # Broadcast to match attention dims
            )
            k_to_cluster = einops.rearrange(
                k_to_cluster,
                "B Q K Klen -> B Klen 1 Q K",
            )
            q_to_cluster = einops.rearrange(
                q_to_cluster,
                "B Q K Qlen -> B 1 Qlen Q K",
            )
            qk_mask = q_to_cluster == k_to_cluster
            # Apply clustering mask using jnp.where
            att_mask = jnp.where(
                use_clustering, jnp.logical_and(causal_mask, qk_mask), causal_mask
            )

            # Select vectors based on cluster assignments
            q_selected = (
                einops.rearrange(q, "B Qlen Q K D -> B Q K 1 Qlen D")
                * cluster_q_alignment_mask
            )
            k_selected = (
                einops.rearrange(k, "B Klen K D -> B 1 K 1 Klen D")
                * cluster_k_alignment_mask
            )

            # Compute alignment scalars
            q_alignment_scalar = einops.reduce(
                q_selected,
                "B Q K n_clusters Qlen D -> ",
                "mean",
            )
            k_alignment_scalar = einops.reduce(
                k_selected,
                "B Q K n_clusters Klen D -> ",
                "mean",
            )


            if h.apply_alibi:
                logits = alibi.apply(logits)

            logits = jnp.where(att_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d Qlen Klen Q K/t, B/d Klen K/t D -> B/d Qlen Q K/t D", probs, v
            )
            w_o = shardops.all_gather(
                "M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_o)
            )
            attn_out = hidden_mult * shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, M Q K/t D -> B/d Qlen M", attn_out, w_o
            )
            attn_out = shardops.psum_scatter("B/d Qlen M -> B/d Qlen M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(
                shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln2))
            )
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_gate)
            )
            gate_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_gate)
            )
            w_up = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_up)
            )
            up_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_up)
            )
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_down)
            )

            ffn_out_mult = (h.d_ff / h.base.d_ff) ** -p.hidden_param_mult
            ffn_out = ffn_out_mult * shardops.einsum_unreduced(
                "B/d L F/t, M F/t -> B/d L M", y, w_down
            )
            ffn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", ffn_out)

            return jnp.bfloat16(x + ffn_out), (
                q_alignment_scalar,
                k_alignment_scalar,
                avg_cluster_alignment,
            )

        x, (q_alignments, k_alignments, avg_cluster_alignments) = jax.lax.scan(
            loop_body, jnp.bfloat16(x), self.transformer
        )

        ##### Final layernorm and output projection.
        x = shardops.all_gather("B/d L M/t -> B/d L M", x)
        ln = shardops.all_gather("M/t/d -> M", jnp.float32(self.final_layer_norm))
        x = jnp.bfloat16(rms_norm(x) * ln)
        unembed = unembed_mult * shardops.all_gather(
            "V/t M/d -> V/t M", jnp.bfloat16(self.unembed)
        )
        logits = shardops.einsum_unreduced(
            "B/d L M, V/t M -> B/d L V/t",
            x,
            unembed,
            preferred_element_type=jnp.float32,
        )

        return logits, (
            q_alignments.sum(),
            k_alignments.sum(),
            avg_cluster_alignments.sum(),
        )

    @typechecked
    def loss(
        self,
        h: Hparams,
        batch: TokenBatch,
        use_clustering: u32[b""],
    ) -> Tuple[f32[b""], Tuple[f32[b""], f32[b""], f32[b""], f32[b""]]]:
        # Given sequence-packed targets:
        #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        # we want inputs:
        #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
        # which we get by shifting the targets right by 1 and
        # masking sequence-start tokens to 0.
        inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
        is_seq_start: bool_[b"batch/d len"] = batch.is_seq_start
        inputs: u32[b"batch/d len"] = jnp.where(is_seq_start, 0, inputs)

        logits, (q_alignments, k_alignments, avg_cluster_alignment) = self.forward_pass(
            h, inputs, is_seq_start, use_clustering
        )
        max_logits: f32[b"batch/d len 1"] = lax.pmax(
            jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t"
        )
        logits = logits - max_logits
        sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
        logsumexp = jnp.log(sum_logits)
        logprobs: f32[b"batch/d len V/t"] = logits - logsumexp
        logprobs_at_targets = shardops.index_unreduced(
            "batch/d len [V/t], batch/d len -> batch/d len", logprobs, batch.targets
        )
        logprobs_at_targets = shardops.psum_scatter(
            "batch/d len -> batch/d len/t", logprobs_at_targets
        )
        tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ("d", "t"))
        ce_loss = -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch)
        alignment_loss = -(q_alignments + k_alignments) / jnp.float32(
            tokens_in_global_batch
        )
        total_loss = ce_loss + alignment_loss
        return total_loss, (
            ce_loss,
            q_alignments,
            k_alignments,
            avg_cluster_alignment,
        )


@pytree_dataclass
class Alibi:
    slopes: f32["K/t"]

    def create(hparams: Hparams) -> "Alibi":
        n_kv = hparams.n_kv
        start = 2.0 ** (-(2.0 ** -(jnp.log2(n_kv) - 3)))
        slopes = start * (start ** shardops.arange("t", n_kv))

        return Alibi(slopes=slopes)

    def apply(self, logits: f32["B/d Qlen Klen Q K/t"]) -> f32["1 Qlen Klen 1 K/t"]:
        Qlen = logits.shape[1]
        slopes = einops.rearrange(self.slopes, "K -> 1 1 1 K")

        position_bias = jnp.arange(Qlen)[None, :] - jnp.arange(Qlen)[:, None]
        position_bias = einops.rearrange(position_bias, "Qlen Klen -> Qlen Klen 1 1")

        bias: f32["1 Qlen Klen 1 K/t"] = einops.rearrange(
            position_bias * slopes, "Qlen Klen 1 K -> 1 Qlen Klen 1 K"
        )
        return logits + bias


@pytree_dataclass
class RopeTable:
    sin: f32["len d_head2"]
    cos: f32["len d_head2"]

    @staticmethod
    def create(max_len: int, hparams: Hparams) -> "RopeTable":
        rope_max_timescale = hparams.rope_max_timescale
        d_head = hparams.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(
            0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False
        )
        position = jnp.arange(max_len, dtype=jnp.int32)
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = einops.rearrange(self.sin, rearrange_spec)
        cos = einops.rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(
    x: Union[
        bf16[b"batch/d len M"],
        bf16[b"batch/d len d_compressed"],
        bf16[b"batch/d len Q K/t D"],
        bf16[b"batch/d len K/t D"],
    ]
) -> Union[
    bf16[b"batch/d len M"],
    bf16[b"batch/d len d_compressed"],
    bf16[b"batch/d len Q K/t D"],
    bf16[b"batch/d len K/t D"],
]:
    mean2 = save_for_backward(
        jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True)
    )
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]
    q_alignments: f32[b""]
    k_alignments: f32[b""]
    avg_cluster_alignment: f32[b""]
    total_loss: f32[b""]


@dataclass(frozen=True)
class TrainingHparams:
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: float
    weight_decay: float
    warmup_steps: int
    steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    learning_rate: float
    tokens: TokenBatchParams
    seed: int
    queue: Optional[str] = None
    use_grad_clip: Optional[bool] = True
    use_gpu: Optional[bool] = False
    use_single_pod: Optional[bool] = False
    use_multistage_training: Optional[bool] = False


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(hparams: Hparams, rng: PRNGKey) -> "State":
        weights = Model.init(hparams, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
@shardtypes.scope
def training_step(
    state: State,
    step: u32[b""],
    h: Hparams,
    hparams: TrainingHparams,
    batch: TokenBatch,
) -> Tuple[Any, Metrics]:
    @partial(
        shardtypes.typed_shard_map, check_rep=False
    )  # check_rep=False for https://github.com/google/jax/issues/20335
    def sharded_step(
        state: State, step: u32[b""], batch: TokenBatch
    ) -> Tuple[State, Metrics]:
        # Determine if we should use clustering based on training progress
        use_clustering = jnp.array(
            step >= int(hparams.steps * h.clustering_start_fraction), dtype=jnp.uint32
        )

        ((loss, (ce_loss, q_alignments, k_alignments, avg_cluster_alignment)), grad) = (
            jax.value_and_grad(
                lambda weights: weights.loss(h, batch, use_clustering), has_aux=True
            )(state.weights)
        )
        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, ("d", "t"))
        ce_loss = jax.lax.psum(ce_loss, ("d", "t"))
        q_alignments = jax.lax.psum(q_alignments, ("d", "t"))
        k_alignments = jax.lax.psum(k_alignments, ("d", "t"))
        avg_cluster_alignment = jax.lax.psum(avg_cluster_alignment, ("d", "t"))

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        warmup_lr = (
            jnp.float32(step) / jnp.float32(hparams.warmup_steps)
        ) * hparams.learning_rate
        cosine = jnp.cos(
            jnp.pi
            * (
                jnp.float32(step - hparams.warmup_steps)
                / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps)
            )
        )
        cosine_lr = hparams.learning_rate * (
            hparams.cosine_learning_rate_final_fraction
            + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        global_norm_square = jnp.float32(0.0)
        for key, g in vars(grad.transformer).items():
            if "clusters" in key:
                continue
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, ("d", "t"))
        global_norm = jnp.sqrt(global_norm_square)

        base = h.base

        p = get_parameterization(h.parameterization)
        target_head_dim = h.n_kv * h.n_q_per_kv * h.d_head
        base_head_dim = base.n_kv * base.n_q_per_kv * base.d_head

        embed_lr_scale = h.gamma_embed * (h.d_model / base.d_model) ** -p.embed_lr
        unembed_lr_scale = h.gamma_unembed * (h.d_model / base.d_model) ** -p.unembed_lr

        lr_scales = Model(
            embed=embed_lr_scale,
            unembed=unembed_lr_scale,
            transformer=Transformer(
                ln1=1.0,
                ln2=1.0,
                ln_q_nope=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
                ln_k_nope=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
                w_q=h.gamma_hidden * (target_head_dim / base_head_dim) ** -p.hidden_lr,
                w_kv=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
                w_o=h.gamma_hidden * (target_head_dim / base_head_dim) ** -p.hidden_lr,
                w_gate=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
                w_up=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
                w_down=h.gamma_hidden * (h.d_ff / base.d_ff) ** -p.hidden_lr,
                clusters=1.0,
            ),
            final_layer_norm=1.0,
        )

        if hparams.use_grad_clip:
            clip_value = 1.0
            rescale = jnp.minimum(1.0, clip_value / global_norm)
        else:
            rescale = 1.0

        new_ps = []
        new_mus = []
        new_nus = []
        for p, g, mu, nu, spec, lr_scale in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
            tree_leaves(shardtypes.make_partition_specs(State)),
            tree_leaves(lr_scales),
        ):
            assert shardtypes.is_fully_sharded(
                spec
            ), "Weight update is only correctly scaled for fully sharded weights."
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1
            mu_hat = mu / (1 - jnp.float32(hparams.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hparams.adam_b2) ** completed_steps)
            # as per C.5. in https://arxiv.org/pdf2407.05872
            # they mention introducing hp a, b to below function,
            # TODO: test and see if a = b = something besides 1
            g = jnp.arctan2(mu_hat, jnp.sqrt(nu_hat))

            # Weight decay
            g += hparams.weight_decay * p
            # Learning rate
            g *= lr * lr_scale

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        metrics = Metrics(
            loss=ce_loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
            q_alignments=q_alignments,
            k_alignments=k_alignments,
            avg_cluster_alignment=avg_cluster_alignment,
            total_loss=loss,
        )
        return new_state, metrics

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
    root_working_dir: str
    model_name: Optional[str]


@dataclass(frozen=True)
class MeshConfig:
    d: int
    t: int


@dataclass(frozen=True)
class Config:
    model: Hparams
    training: TrainingHparams
    paths: Paths
    num_hosts: int
    checkpoint_interval: int
    mesh: MeshConfig
    io: training_io.IOConfig
    flat_tokens: Optional[FlatTokensParams] = None
    hf_dataset: Optional[HuggingFaceDataParams] = None

    def __post_init__(self):
        assert (
            self.flat_tokens is not None or self.hf_dataset is not None
        ), "Must provide either flat_tokens or hf_dataset."
        assert not (
            self.flat_tokens is not None and self.hf_dataset is not None
        ), "Should not specify both flat_tokens and hf_dataset."

    @cached_property
    def training_data(self) -> Union[FlatTokensParams, HuggingFaceDataParams]:
        return self.flat_tokens or self.hf_dataset


def main_contained(config, logger):
    """Main program, which does not access external services except as specified by config.paths or logger."""
    # Use partitionable (and hopefully fusable!) RNG.
    #
    # This is slower in compute time than 'unsafe_rbg' with flag '--xla_tpu_spmd_rng_bit_generator_unsafe=true',
    # but hopefully faster in memory time because it's fusable.
    # TODO: check this is true and if not, provide our own that actually is fusable.

    # 4x 1 chip (2 cores) per process:
    if config.training.use_single_pod:
        os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
        os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    jax.config.update("jax_threefry_partitionable", True)
    with Mesh(
        mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()),
        ("d", "t"),
    ):
        root_rng = jax.random.PRNGKey(config.training.seed)

        loader = get_loader("train", config.training_data, config.training.tokens)
        assert (
            config.model.vocab > loader.max_token_id
        ), f"{config.model.vocab} vs {loader.max_token_id}"
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        model_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        model_dir = os.path.join(config.paths.root_working_dir, model_name)
        print(model_name)
        training_io.mkdir(model_dir)
        state = jax.jit(partial(State.init, config.model))(
            fold_in_str(root_rng, "init")
        )

        state, start_step = training_io.load_checkpoint_if_it_exists(
            model_dir, state, config.io
        )

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, jnp.uint32(0), config.model, config.training, loader.load(0)
        ).compile()
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # training_io.save_hlo_svg(os.path.join(model_dir, f'training_step_optimized_hlo_{date}.svg'), c_training_step)

        log_interval = math.ceil(config.training.steps / 5000)
        print(f"{log_interval=}")

        cum_metrics = None

        def update_metrics(metrics: Metrics):
            nonlocal cum_metrics
            cum_metrics.loss += metrics.loss
            cum_metrics.avg_cluster_alignment += metrics.avg_cluster_alignment
            cum_metrics.grad_norm += metrics.grad_norm
            cum_metrics.raw_grad_norm += metrics.raw_grad_norm
            cum_metrics.learning_rate += metrics.learning_rate
            cum_metrics.q_alignments += metrics.q_alignments
            cum_metrics.k_alignments += metrics.k_alignments
            cum_metrics.total_loss += metrics.total_loss

        start_time = time.time()

        for step in range(start_step, config.training.steps):
            if step % config.checkpoint_interval == 0 and step > start_step:
                training_io.save_checkpoint(model_dir, step, state, config.io)

            # We profile on the second step, because the first step has a long pause for XLA
            # compilation and initial shuffle buffer loading.
            if training_io.is_device_0() and step == start_step + 1:
                jax.block_until_ready(state)
                training_io.start_profile()
                profile_start = time.time()

            state, output = c_training_step(state, jnp.uint32(step), loader.load(step))

            # if half way point and multistage training is enabled, double seq length and halve batch size
            if (
                step == config.training.steps // 2
                and config.training.use_multistage_training
            ):
                print("updating seq length and batch size")
                tokens = replace(
                    config.training.tokens,
                    len=config.training.tokens.len * 2,
                    batch=max(config.mesh.d, config.training.tokens.batch // 2),
                )
                config = replace(
                    config, training=replace(config.training, tokens=tokens)
                )
                loader = get_loader(
                    "train", config.training_data, config.training.tokens
                )
                c_training_step = training_step.lower(
                    state,
                    jnp.uint32(0),
                    config.model,
                    config.training,
                    loader.load(step),
                ).compile()

            batch = loader.load(step)
            state, output = c_training_step(state, jnp.uint32(step), batch)

            # Run profile for two steps, to include data loading time in between them.
            if training_io.is_device_0() and step == start_step + 2:
                jax.block_until_ready(state)
                profile_duration = time.time() - profile_start
                training_io.stop_profile(model_dir)

                # Print MFU, including (one step of) data loading time.
                print(f"Profile time: {profile_duration}s for 2 steps.")
                model_params = jax.tree.reduce(
                    operator.add, jax.tree.map(lambda w: w.size, state.weights)
                )
                tokens = loader.load(step).targets.size
                print(f"Model params: {model_params:_}")
                print(f"Tokens: {tokens:_}")
                device_flops = training_io.get_flops_per_device()
                num_devices = jax.device_count()
                print(
                    f"MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU"
                )

            if step % log_interval == 0:
                if cum_metrics:
                    cum_metrics = Metrics(
                        loss=cum_metrics.loss / log_interval,
                        avg_cluster_alignment=cum_metrics.avg_cluster_alignment
                        / log_interval,
                        learning_rate=cum_metrics.learning_rate / log_interval,
                        grad_norm=cum_metrics.grad_norm / log_interval,
                        raw_grad_norm=cum_metrics.raw_grad_norm / log_interval,
                        q_alignments=cum_metrics.q_alignments / log_interval,
                        k_alignments=cum_metrics.k_alignments / log_interval,
                        total_loss=cum_metrics.total_loss / log_interval,
                    )
                else:
                    cum_metrics = output
                training_io.log(step, logger, cum_metrics)
                cum_metrics = output
            else:
                update_metrics(output)

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")


def clear_tpu_locks():
    try:
        raw_pids = subprocess.run(
            ["lsof", "-w", "/dev/accel0"], capture_output=True, text=True
        ).stdout
        pids = set()
        for line in raw_pids.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 1:
                pids.add(parts[1])
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            os.remove("/tmp/libtpu_lockfile")
    except Exception as e:
        print(f"Error clearing TPU locks: {e}")
        pass


def get_filtered_overrides():
    """Get filtered override strings from Hydra config, excluding certain overrides."""
    overrides = hydra.core.hydra_config.HydraConfig.get()["job"]["override_dirname"]
    ignore_overrides = [
        "training.queue",
    ]
    return [
        f"{override.lstrip('+').split('=')[0].split('.')[-1]}={override.split('=')[1]}"
        for override in overrides.split(",")
        if override and override.lstrip("+").split("=")[0] not in ignore_overrides
    ]


def get_model_name(config_name: str):
    overrides = get_filtered_overrides()
    overrides = "_".join(overrides)
    return f"{config_name}_{overrides}" if overrides else config_name


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    if config.training.queue:
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        task_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        git_branch_name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        task = Task.init(
            project_name=f"{config_name}/{git_branch_name}", task_name=task_name
        )

        # Add git branch and filtered overrides as tags
        override_tags = get_filtered_overrides()
        task.add_tags([git_branch_name] + override_tags)

        if config.training.use_gpu:
            task.set_packages("requirements-gpu.txt")
        else:
            task.set_packages("requirements-tpu.txt")

        result = subprocess.run(
            ["datasets-cli", "env"], capture_output=True, text=True, check=True
        )

        print("Datasets CLI Environment:")
        print(result.stdout)

        logger = task.get_logger()
        task.execute_remotely(queue_name=config.training.queue)
        task.launch_multi_node(
            config.num_hosts, wait=True, queue=config.training.queue + "-workers"
        )
        clear_tpu_locks()
        jax.distributed.initialize(
            os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"]),
        )
    else:
        logger = None
    main_contained(config, logger)

    if not training_io.is_device_0():
        task.set_system_tags((task.get_system_tags() or []) + ["hidden"])


if __name__ == "__main__":
    main()
