# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    attn_window: int | None = None
    latent_token_id: int | None = None


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def rope_with_positions(self, v, positions):
        freqs = self.freqs
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        r_phases = positions.view(1, 1, -1, 1) * freqs
        return self.rope(r_phases, v)

    def forward(self, Q, K, V, token_types=None, segment_ids=None):
        freqs = self.freqs
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        assert K is Q
        B, nh, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=freqs.device,
                dtype=freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        if token_types is not None:
            if segment_ids is None:
                raise ValueError("segment_ids must be provided when token_types is set.")
            return self._forward_segmented(QR, KR, V, token_types, segment_ids)

        attn_window = getattr(self.config, "attn_window", None)
        if attn_window is None or attn_window >= T:
            # Current attention
            scores = (QR @ KR.mT).tril(diagonal=-1)
            return scores @ V
        if attn_window < 1:
            raise ValueError("attn_window must be >= 1")

        D = V.size(-1)
        out = torch.zeros((B, nh, T, D), device=V.device, dtype=V.dtype)
        block = min(attn_window, T)
        positions = torch.arange(T, device=Q.device)

        for start in range(0, T, block):
            end = min(start + block, T)
            k_start = max(0, start - attn_window)
            q = QR[:, :, start:end, :]
            k = KR[:, :, k_start:end, :]
            v = V[:, :, k_start:end, :]

            scores = q @ k.mT
            q_pos = positions[start:end].view(1, 1, -1, 1)
            k_pos = positions[k_start:end].view(1, 1, 1, -1)
            mask = (k_pos < q_pos) & (k_pos >= q_pos - attn_window)
            scores = scores * mask

            out[:, :, start:end, :] = scores @ v

        return out

    def _attend(self, QR, KR, V, q_idx, k_idx, strict_causal=True):
        q = QR[:, :, q_idx, :]
        k = KR[:, :, k_idx, :]
        v = V[:, :, k_idx, :]
        scores = q @ k.mT
        if strict_causal:
            q_pos = q_idx.view(-1, 1)
            k_pos = k_idx.view(1, -1)
            mask = k_pos < q_pos
            scores = scores * mask
        return scores @ v

    def _forward_segmented(self, QR, KR, V, token_types, segment_ids):
        B, nh, T, _ = QR.size()
        if B != 1:
            raise ValueError("Segmented attention only supports batch size 1.")

        token_types = token_types[0]
        segment_ids = segment_ids[0]
        D = V.size(-1)
        out = torch.zeros((B, nh, T, D), device=V.device, dtype=V.dtype)

        type_context = token_types == 0
        type_latent = token_types == 1
        type_prompt = token_types == 2
        all_latent_idx = type_latent.nonzero(as_tuple=False).squeeze(-1)
        segments = torch.unique(segment_ids)

        for seg in segments.tolist():
            seg_mask = segment_ids == seg
            seg_context_idx = (seg_mask & type_context).nonzero(as_tuple=False).squeeze(-1)
            seg_prompt_idx = (seg_mask & type_prompt).nonzero(as_tuple=False).squeeze(-1)
            seg_latent_idx = (seg_mask & type_latent).nonzero(as_tuple=False).squeeze(-1)

            if seg_context_idx.numel():
                if seg_latent_idx.numel():
                    k_idx = torch.cat([seg_context_idx, seg_latent_idx], dim=0)
                else:
                    k_idx = seg_context_idx
                out[:, :, seg_context_idx, :] = self._attend(
                    QR, KR, V, seg_context_idx, k_idx, strict_causal=True
                )

            if seg_prompt_idx.numel():
                key_parts = [seg_context_idx, seg_prompt_idx]
                if all_latent_idx.numel():
                    key_parts.append(all_latent_idx)
                k_idx = torch.cat([idx for idx in key_parts if idx.numel()], dim=0)
                out[:, :, seg_prompt_idx, :] = self._attend(
                    QR, KR, V, seg_prompt_idx, k_idx, strict_causal=True
                )

            if seg_latent_idx.numel():
                if all_latent_idx.numel():
                    k_idx = torch.cat([seg_context_idx, all_latent_idx], dim=0)
                else:
                    k_idx = seg_context_idx
                out[:, :, seg_latent_idx, :] = self._attend(
                    QR, KR, V, seg_latent_idx, k_idx, strict_causal=True
                )

        return out


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        token_types=None,
        segment_ids=None,
        grad_mask=None,
        cache_indices=None,
        return_cache=False,
    ):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        if grad_mask is not None:
            grad_mask = grad_mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)
            x = x * grad_mask + x.detach() * (1 - grad_mask)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        caches = [] if return_cache else None
        positions = None
        if return_cache:
            if cache_indices is None:
                raise ValueError("cache_indices is required when return_cache is set.")
            positions = torch.arange(
                T, device=x.device, dtype=self.attn.freqs.dtype
            )

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                token_types=token_types,
                segment_ids=segment_ids,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

            if return_cache:
                cache_pos = positions[cache_indices]
                k_cache = self.attn.rope_with_positions(
                    x_sparse[:, :, cache_indices, :], cache_pos
                )
                v_cache = x[:, :, cache_indices, :]
                caches.append({"k": k_cache, "v": v_cache})

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_cache:
            return logits, loss, caches
        return logits, loss

    def forward_with_cache(self, idx, cache, grad_mask=None, position_offset=0):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        if grad_mask is not None:
            grad_mask = grad_mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)
            x = x * grad_mask + x.detach() * (1 - grad_mask)
        x = self.ln(x)

        positions = (
            torch.arange(T, device=x.device, dtype=self.attn.freqs.dtype)
            + position_offset
        )
        causal = torch.tril(
            torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=-1
        )

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            q = self.attn.rope_with_positions(x_sparse, positions)
            k_prompt = q
            v_prompt = x

            k_ctx = cache[level]["k"]
            v_ctx = cache[level]["v"]

            scores_ctx = q @ k_ctx.mT
            scores_prompt = q @ k_prompt.mT
            scores_prompt = scores_prompt * causal
            yKV = scores_ctx @ v_ctx + scores_prompt @ v_prompt

            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
