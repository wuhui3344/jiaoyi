"""Kronos 的 ONNX 导出包装。

为什么需要这一层
================
官方 ``KronosPredictor`` 在自回归解码的每一步都会把**整个上下文窗口**重新前向一次::

    s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], stamp)

其中 ``input_tokens`` 的长度是 ``min(L + i, max_context)``。以 Kronos-small
(``d_model=512, n_layers=8, ff_dim=1024``) 为例, ``lookback=512, pred_len=120``
时总计算量约 **1.7 TFLOPs**, 在手机 CPU 上需要数分钟, 完全不可用。

本模块把同一套权重重新组织成「一次 prefill + 每步单 token 增量解码」:

======================  ==========================================================
图                      输入 / 输出
======================  ==========================================================
``tokenizer_encode``    ``x[B,L,6]`` -> ``s1_ids[B,L], s2_ids[B,L]``
``tokenizer_decode``    ``s1_ids[B,T], s2_ids[B,T]`` -> ``y[B,T,6]``
``kronos_decode``       ``s1_ids[B,T], s2_ids[B,T], stamp[B,T,5], t_k/nl,B,H,P,hd],``
                        ``t_v, d_k[B,dh,P,dhd], d_v``
                        -> ``s1_logits[B,1024]``, ``context[B,512]``, 更新后的 KV cache
``kronos_step_s2``      ``s1_id[B,1], context[B,512], d_k, d_v`` -> ``s2_logits[B,1024]``
======================  ==========================================================

``T = L, P = 0`` 即 prefill, ``T = 1, P > 0`` 即增量步 —— 两者共用同一张解码图。

数学上的等价性
==============
官方实现每步都重新计算整个窗口, 但**只读取最后一个位置的 logits**。由于注意力是
因果的, 且 RoPE 的位置是「窗口内相对位置」, 对于不发生滑动的情况
(``lookback + pred_len <= max_context``), 第 ``t`` 个 token 的隐藏状态只依赖
``0..t``, 一旦算出就不会再变。因此增量解码与官方实现逐位等价。

另外两个必须复刻的实现细节:

1. **KV cache 必须存未旋转的 key**。滑动窗口或位置变化时, RoPE 会失效, 所以缓存
   ``k_proj`` 的原始输出, 每次使用时按当前相对位置重新施加 RoPE。
   (本项目通过 ``lookback + pred_len <= 512`` 的不变式彻底避免了滑动。)

2. **依赖感知层 (DependencyAwareLayer) 的 RoPE 是一个"怪癖"**。
   ``MultiHeadCrossAttentionWithRoPE.forward`` 用 ``q.shape[-2]`` 构造 cos/sin 表,
   然后同时作用在 q 和 k 上。在 ``decode_s2`` 中 ``q_len`` 恒为 1, 因此 k 实际是
   在**位置 0** 被旋转的 (广播)。本模块完整复刻该行为, 否则输出会与官方实现不符。

约束
====
* ``lookback + pred_len <= max_context`` (Kronos-small 为 512)。否则官方实现会启用
  滑动窗口, 滑动会改变已缓存 token 的相对位置编码, 增量解码不再等价。Kotlin 侧强制
  该约束 (必要时自动截断 ``lookback``)。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_inv_freq",
    "TokenizerEncodeONNX",
    "TokenizerDecodeONNX",
    "KronosDecodeONNX",
    "KronosStepS2ONNX",
    "IncrementalReferenceDecoder",
]


# --------------------------------------------------------------------------------------
# 基础算子 (与 model/module.py 的语义逐行对应)
# --------------------------------------------------------------------------------------


def build_inv_freq(head_dim: int) -> torch.Tensor:
    """复刻 ``RotaryPositionalEmbedding.__init__`` 的 ``inv_freq``。

    ``inv_freq = 1.0 / (10000 ** (arange(0, dim, 2).float() / dim))``
    """
    return 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))


def _rope_tables(
    inv_freq: torch.Tensor, position: torch.Tensor, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """由位置编号构造 cos/sin 表。

    对应 ``RotaryPositionalEmbedding._update_cos_sin_cache``::

        t = arange(seq_len)
        freqs = einsum('i,j->ij', t, inv_freq)
        emb = cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]

    Args:
        inv_freq: shape ``[head_dim // 2]``
        position: shape ``[B, T]``, int64

    Returns:
        ``(cos, sin)`` 各为 ``[B, 1, T, head_dim]``, 便于对 ``[B, H, T, head_dim]`` 广播。
    """
    freqs = position.to(torch.float32).unsqueeze(-1) * inv_freq.to(torch.float32)  # [B,T,hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B,T,hd]
    return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """复刻 ``RotaryPositionalEmbedding._rotate_half``。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """复刻 ``RotaryPositionalEmbedding.forward`` (含广播行为)。"""
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """``[B,T,d]`` -> ``[B,H,T,head_dim]`` (复刻 ``.view().transpose()``)。"""
    b, t, _ = x.shape
    return x.reshape(b, t, n_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """``[B,H,T,head_dim]`` -> ``[B,T,d]`` (复刻 ``.transpose().contiguous().view()``)。"""
    b, h, t, hd = x.shape
    return x.transpose(1, 2).reshape(b, t, d_model)


def _manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_len: torch.Tensor | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """手写 scaled dot-product attention。

    不使用 ``F.scaled_dot_product_attention``, 因为手写版本在导出 ONNX 时更稳定,
    且与 float32 参考实现的一致性更好。

    本函数的 ``k`` / ``v`` 有两种形态:

    * **拼接形态** (主 Transformer): ``k = cat([定长历史缓存, 新 token 的 key], dim=2)``,
      总长度 ``C + T``。历史有效区间是 ``[0, past_len)``, 新 token 落在 ``[C, C + T)``。
      —— 注意新 token 的起始下标是**容量 C** 而不是 ``past_len``, 这正是掩码必须
      区分"历史槽位"与"新增槽位"的原因。
    * **连续形态** (依赖层): ``k`` 就是定长缓存本身, 有效区间即 ``[0, past_len)``。

    Args:
        past_len: 有效 key 数量 (标量张量)。``None`` 表示全部有效。
        causal: 是否施加因果掩码。
    """
    head_dim = q.shape[-1]
    att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
    if past_len is not None:
        q_len, k_len = q.shape[-2], k.shape[-2]
        cached_len = k_len - q_len  # 历史缓存部分长度 (拼接形态下 = 容量 C; 连续形态下 = 0)
        kj = torch.arange(k_len, device=q.device).unsqueeze(0)  # [1, k_len]
        if causal and cached_len > 0:
            qi = torch.arange(q_len, device=q.device).unsqueeze(-1)  # [q_len, 1]
            is_new = kj >= cached_len  # [1, k_len]
            new_ok = (kj - cached_len) <= qi  # [q_len, k_len]
            hist_ok = kj < past_len  # 广播到 [q_len, k_len]
            # 用 And/Or/Not 而不是 torch.where: ONNX Runtime 尚未实现 bool 类型的 Where
            keep = (new_ok & is_new) | (hist_ok & ~is_new)
        elif causal:
            qi = torch.arange(q_len, device=q.device).unsqueeze(-1)
            keep = kj <= qi  # 纯因果, 无历史
        else:
            keep = kj < past_len
        att = att.masked_fill(~keep, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return torch.matmul(att, v)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """复刻 ``RMSNorm.forward``。"""
    dtype = x.dtype
    out = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return out.to(dtype) * weight


# --------------------------------------------------------------------------------------
# Tokenizer 图
# --------------------------------------------------------------------------------------


class TokenizerEncodeONNX(nn.Module):
    """``KronosTokenizer.encode(x, half=True)`` 的等价图。

    数据流 (对应 ``KronosTokenizer.encode`` + ``BSQuantizer.forward``)::

        z = embed(x) -> 3 x TransformerBlock -> quant_embed
        z = F.normalize(z, dim=-1)
        zq = sign(z)                     # BinarySphericalQuantizer.quantize
        s1_ids = sum_{i<10}  [zq_i >= 0] * 2^i     # 低 10 位
        s2_ids = sum_{i<10}  [zq_{10+i} >= 0] * 2^i

    注意 ``q_scale = 1/sqrt(codebook_dim)`` 只是整体缩放, 不影响 ``>= 0`` 的符号判断,
    故此处省略。
    """

    def __init__(self, tokenizer, s1_bits: int, s2_bits: int):
        super().__init__()
        self.tok = tokenizer
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.register_buffer("s1_weights", 2 ** torch.arange(s1_bits, dtype=torch.int64))
        self.register_buffer("s2_weights", 2 ** torch.arange(s2_bits, dtype=torch.int64))

    @staticmethod
    def _bits_to_index(bit_flags: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """复刻 ``BSQuantizer.bits_to_indices``。"""
        return (bit_flags.to(torch.int64) * weights).sum(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.tok.embed(x)
        for layer in self.tok.encoder:
            z = layer(z)
        z = self.tok.quant_embed(z)
        z = F.normalize(z, dim=-1)
        # BinarySphericalQuantizer.quantize: zhat = where(z > 0, 1, -1)  (STE 的梯度部分不影响前向)
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        s1_flags = zhat[..., : self.s1_bits] >= 0
        s2_flags = zhat[..., self.s1_bits :] >= 0
        return self._bits_to_index(s1_flags, self.s1_weights), self._bits_to_index(s2_flags, self.s2_weights)


class TokenizerDecodeONNX(nn.Module):
    """``KronosTokenizer.decode(x, half=True)`` 的等价图。

    数据流 (对应 ``indices_to_bits`` + ``KronosTokenizer.decode``)::

        bit_i(s1) = (s1_id >> i) & 1   for i in 0..9    -> +1 / -1
        bit_i(s2) = (s2_id >> i) & 1   for i in 0..9    -> +1 / -1
        codes = cat([s1_bits, s2_bits], -1) * (1 / sqrt(20))
        y = head(3 x TransformerBlock(post_quant_embed(codes)))
    """

    def __init__(self, tokenizer, s1_bits: int, s2_bits: int):
        super().__init__()
        self.tok = tokenizer
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        codebook_dim = s1_bits + s2_bits
        self.q_scale = 1.0 / math.sqrt(codebook_dim)
        self.register_buffer("s1_masks", 2 ** torch.arange(s1_bits, dtype=torch.int64))

    def _indices_to_bits(self, indices: torch.Tensor, n_bits: int) -> torch.Tensor:
        """按位展开为 ±1。

        官方用 ``torch.bitwise_and``, 但该算子需要 ONNX opset 18。这里改用
        ``floor(idx / 2^i) % 2``, 语义完全等价且 opset 17 即可导出, 移动端兼容性更好。
        """
        masks = self.s1_masks[:n_bits]
        shifted = torch.div(indices.to(torch.int64).unsqueeze(-1), masks, rounding_mode="floor")
        bits = torch.remainder(shifted, 2).to(torch.float32)
        return bits * 2.0 - 1.0

    def forward(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor) -> torch.Tensor:
        b1 = self._indices_to_bits(s1_ids, self.s1_bits)
        b2 = self._indices_to_bits(s2_ids, self.s2_bits)
        codes = torch.cat([b1, b2], dim=-1) * self.q_scale
        z = self.tok.post_quant_embed(codes)
        for layer in self.tok.decoder:
            z = layer(z)
        return self.tok.head(z)


# --------------------------------------------------------------------------------------
# Kronos 主模型: 共用的前向构件
# --------------------------------------------------------------------------------------


class _KronosCommon(nn.Module):
    """存放 Kronos 权重引用与共用算子。"""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.s1_bits = model.s1_bits
        self.s2_bits = model.s2_bits
        self.n_layers = model.n_layers
        self.n_heads = model.n_heads
        self.d_model = model.d_model
        self.head_dim = model.d_model // model.n_heads
        self.n_dep_heads = model.dep_layer.cross_attn.n_heads
        self.dep_head_dim = model.dep_layer.cross_attn.head_dim
        # HierarchicalEmbedding.forward 内部对 emb_s1/emb_s2 缩放了 sqrt(d_model),
        # 但 emb_s1 单独使用时 (依赖感知层) 官方并未缩放。这里保留常量以备参考。
        self.emb_scale = math.sqrt(model.embedding.d_model)
        self.register_buffer("t_inv_freq", build_inv_freq(self.head_dim))
        self.register_buffer("d_inv_freq", build_inv_freq(self.dep_head_dim))

    # ----- 输入嵌入 ---------------------------------------------------------------

    def _embed(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
        """复刻 ``Kronos.forward`` 开头的嵌入部分 (eval 模式下 dropout 为恒等)。"""
        x = self.m.embedding([s1_ids.clone(), s2_ids.clone()])
        x = x + self.m.time_emb(stamp.to(torch.int64))
        return self.m.token_drop(x)

    # ----- Transformer block (带 KV cache) ---------------------------------------

    def _block_forward(
        self,
        x: torch.Tensor,
        layer,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        past_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单个 ``TransformerBlock`` 的前向, 支持拼接 KV cache。

        对应 ``TransformerBlock.forward`` (pre-norm 残差结构)::

            residual = x
            x = norm1(x)
            attn = self_attn(x)
            x = residual + attn
            residual = x
            x = norm2(x)
            x = residual + ffn(x)

        ``past_k`` / ``past_v`` 存的是**未施加 RoPE**的投影结果, 形状
        ``[B, n_heads, C, head_dim]`` (``C`` 为预分配容量, 定长)。

        Returns:
            ``(x, k_new, v_new)`` —— ``k_new`` 是**新增**的未旋转 key (长度 ``T``),
            由调用方写入定长缓存。
        """
        residual = x
        h = _rms_norm(x, layer.norm1.weight)

        attn_mod = layer.self_attn
        q = _split_heads(attn_mod.q_proj(h), self.n_heads, self.head_dim)
        k_new = _split_heads(attn_mod.k_proj(h), self.n_heads, self.head_dim)
        v_new = _split_heads(attn_mod.v_proj(h), self.n_heads, self.head_dim)

        q_rot, k_rot = _apply_rope(q, k_new, cos, sin)

        # 缓存里存的是**未旋转**的 key, 必须按它们各自的相对位置补上 RoPE,
        # 否则历史 key 会缺少位置信息 (增量解码最容易踩的坑)。
        # 超出 valid_len 的槽位是尚未写入的垃圾值, 会被掩码屏蔽, 旋转结果无关紧要。
        cache_len = past_k.shape[2]
        pos_past = torch.arange(cache_len, device=past_k.device).unsqueeze(0)
        cos_past, sin_past = _rope_tables(self.t_inv_freq, pos_past, q.dtype)
        past_k_rot = (past_k * cos_past) + (_rotate_half(past_k) * sin_past)

        # 拼接后总长度为 C + T (定长), 有效区间由 valid_len 掩码界定。
        # causal 掩码给出 keep[j] = j <= valid_len - T + i, 恰好等价于
        # "历史位置 j 有效且 j <= 当前位置"。
        k_attn = torch.cat([past_k_rot, k_rot], dim=2)
        v_attn = torch.cat([past_v, v_new], dim=2)

        out = _manual_attention(q_rot, k_attn, v_attn, past_len=past_len, causal=True)
        out = attn_mod.out_proj(_merge_heads(out, self.d_model))
        out = attn_mod.resid_dropout(out)
        x = residual + out

        residual = x
        h = _rms_norm(x, layer.norm2.weight)
        x = residual + layer.ffn(h)
        return x, k_new, v_new

# --------------------------------------------------------------------------------------
# Kronos 解码图
# --------------------------------------------------------------------------------------


class KronosDecodeONNX(_KronosCommon):
    """统一的解码图: 接收 ``T`` 个新 token 与已有 KV cache, 输出 s1 logits 与更新后的 cache。

    同一个图同时覆盖官方实现的两种调用形态:

    ============  ==================  ============================================
    调用          输入形状             对应官方代码
    ============  ==================  ============================================
    prefill       ``T = L, P = 0``     ``decode_s1(prompt)`` 这一步
    增量步         ``T = 1, P >= L``    循环中的每一步 ``decode_s1``
    ============  ==================  ============================================

    之所以能共用一张图: 因果掩码写作 ``keep[j] = (j <= P + i)``, ``P`` 为缓存长度。
    ``T = 1`` 时该式恒成立 (退化为"全部可见"), ``P = 0, T = L`` 时正好是标准三角因果掩码。
    位置编码统一取 ``P + arange(T)`` (窗口内相对位置)。

    依赖层 (DependencyAwareLayer) 的 KV 与主 Transformer 的 KV **始终覆盖相同的位置集合**,
    因此这里同步追加, 不需要任何 off-by-one 补偿。
    """

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor,
        past_len: torch.Tensor,
        t_k: torch.Tensor,
        t_v: torch.Tensor,
    ):
        b, t = s1_ids.shape

        # 窗口内相对位置: 已有 valid_len 个位置, 新 token 依次落在 valid_len..valid_len+T-1
        position = (past_len + torch.arange(t, dtype=torch.int64, device=s1_ids.device)).unsqueeze(0).expand(b, -1)
        cos, sin = _rope_tables(self.t_inv_freq, position, torch.float32)

        x = self._embed(s1_ids, s2_ids, stamp)

        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for layer, pk, pv in zip(self.m.transformer, torch.unbind(t_k, dim=0), torch.unbind(t_v, dim=0)):
            x, k, v = self._block_forward(x, layer, cos, sin, pk, pv, past_len)
            ks.append(k)
            vs.append(v)

        x = _rms_norm(x, self.m.norm.weight)
        s1_logits = self.m.head(x)

        # 依赖层缓存与主缓存同步增长 (两者覆盖同一批位置)。
        # 只输出**新增**的 T 个位置, 由调用方写入定长缓存, 避免每步重新分配。
        dep = self.m.dep_layer
        d_k_new = dep.cross_attn.k_proj(x).reshape(b, t, self.n_dep_heads, self.dep_head_dim).transpose(1, 2)
        d_v_new = dep.cross_attn.v_proj(x).reshape(b, t, self.n_dep_heads, self.dep_head_dim).transpose(1, 2)

        return (
            s1_logits[:, -1, :],
            x[:, -1, :],
            torch.stack(ks, dim=0),
            torch.stack(vs, dim=0),
            d_k_new,
            d_v_new,
        )


class KronosStepS2ONNX(_KronosCommon):
    """s2 logits 头: 输入已采样的 s1 与依赖层 KV cache, 输出 s2 logits。

    ``d_k`` / ``d_v`` 由 ``KronosDecodeONNX`` 维护, 已经覆盖了官方 ``decode_s2`` 所需的
    全部 context 位置, 因此这里**只做注意力, 不再追加**。
    """

    def forward(
        self,
        s1_id: torch.Tensor,
        context_last: torch.Tensor,
        valid_len: torch.Tensor,
        d_k: torch.Tensor,
        d_v: torch.Tensor,
    ):
        b = context_last.shape[0]
        dep = self.m.dep_layer

        # 注意: 官方 ``decode_s2`` 用的是**未缩放**的 ``emb_s1``。
        # sqrt(d_model) 缩放只发生在 ``HierarchicalEmbedding.forward`` 内部。
        sib = self.m.embedding.emb_s1(s1_id)  # [B,1,d]
        q = _split_heads(dep.cross_attn.q_proj(sib), self.n_dep_heads, self.dep_head_dim)

        # 官方怪癖: cos/sin 表长度由 q_len 决定 (decode_s2 中恒为 1),
        # 作用于 k 时广播 -> 所有 key 都在位置 0 被旋转。必须原样复刻。
        pos = torch.zeros(b, 1, dtype=torch.int64, device=context_last.device)
        cos, sin = _rope_tables(self.d_inv_freq, pos, q.dtype)
        q_rot, d_k_rot = _apply_rope(q, d_k, cos, sin)

        # 非因果的全注意力; d_k/d_v 是定长缓存, 用 valid_len 屏蔽未写入的槽位
        out = _manual_attention(q_rot, d_k_rot, d_v, past_len=valid_len, causal=False)
        attn_out = dep.cross_attn.out_proj(_merge_heads(out, self.d_model))

        # 官方实现: return self.norm(hidden_states + attn_out)
        # hidden_states 长度为 T, attn_out 长度为 1 -> 广播, 只取最后一个位置。
        # 这里只传了最后一个位置的 context, 因此等价于取 [-1]。
        merged = dep.norm((context_last + attn_out[:, -1, :]).unsqueeze(1))  # [B,1,d]
        s2_logits = self.m.head.cond_forward(merged)
        return s2_logits[:, -1, :]


# --------------------------------------------------------------------------------------
# 纯 PyTorch 的增量解码参考实现 (用于在导出前自检等价性)
# --------------------------------------------------------------------------------------


class IncrementalReferenceDecoder:
    """用 PyTorch 复刻 Kotlin 侧将要执行的控制流, 便于与官方实现做数值对拍。

    用法::

        dec = IncrementalReferenceDecoder(model, tokenizer, max_context=512)
        preds = dec.generate(x_norm, x_stamp, y_stamp, pred_len=8, sample_count=1, top_k=1, top_p=1.0)
    """

    def __init__(self, model, tokenizer, max_context: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context

        self.decoder = KronosDecodeONNX(model).eval()
        self.step_s2 = KronosStepS2ONNX(model).eval()
        self.enc = TokenizerEncodeONNX(tokenizer, model.s1_bits, model.s2_bits).eval()
        self.dec = TokenizerDecodeONNX(tokenizer, model.s1_bits, model.s2_bits).eval()

    # -- 采样: 复刻 kronos.sample_from_logits / top_k_top_p_filtering ----------------

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        logits = logits / temperature
        if top_k > 0 and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        elif top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            scatter = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
            logits = logits.masked_fill(scatter, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        x_stamp: torch.Tensor,
        y_stamp: torch.Tensor,
        pred_len: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> torch.Tensor:
        """返回 ``[sample_count, pred_len, 6]`` (未反归一化)。"""
        lookback = x.shape[1]
        assert lookback + pred_len <= self.max_context, (
            f"lookback({lookback}) + pred_len({pred_len}) 必须 <= max_context({self.max_context})"
        )

        # tokenizer 是确定性的: 只编码一次再广播, 与官方"先 repeat 再 encode"等价
        s1_ids, s2_ids = self.enc(x)
        s1_ids = s1_ids.repeat(sample_count, 1)
        s2_ids = s2_ids.repeat(sample_count, 1)

        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)  # [1, L+P, 5]

        # 预分配定长 KV cache, 容量 = lookback + pred_len (<= max_context)。
        # 定长带来两个好处: 图中所有形状静态 (ORT 更快), 且整个解码过程零内存分配。
        nl, nh, hd = self.model.n_layers, self.model.n_heads, self.model.d_model // self.model.n_heads
        dep = self.model.dep_layer.cross_attn
        b = sample_count
        capacity = lookback + pred_len

        t_k = torch.zeros(nl, b, nh, capacity, hd, dtype=x.dtype)
        t_v = torch.zeros_like(t_k)
        d_k = torch.zeros(b, dep.n_heads, capacity, dep.head_dim, dtype=x.dtype)
        d_v = torch.zeros_like(d_k)
        valid = 0

        def decode(s1_in: torch.Tensor, s2_in: torch.Tensor, stamp_in: torch.Tensor):
            """写入新位置并返回最后位置的 s1 logits 与 context。"""
            nonlocal t_k, t_v, d_k, d_v, valid
            plen = torch.tensor([valid], dtype=torch.int64)
            logits, ctx, k_new, v_new, dk_new, dv_new = self.decoder(s1_in, s2_in, stamp_in, plen, t_k, t_v)
            span = s1_in.shape[1]
            t_k[:, :, :, valid : valid + span, :] = k_new
            t_v[:, :, :, valid : valid + span, :] = v_new
            d_k[:, :, valid : valid + span, :] = dk_new
            d_v[:, :, valid : valid + span, :] = dv_new
            valid += span
            return logits, ctx

        def s2_head(s1_in: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            return self.step_s2(s1_in, ctx, torch.tensor([valid], dtype=torch.int64), d_k, d_v)

        # ---- prefill: T = L, 写入位置 0..L-1 ----
        s1_logits, ctx = decode(s1_ids, s2_ids, full_stamp[:, :lookback, :])
        s1_cur = self.sample(s1_logits, temperature, top_k, top_p)
        s2_cur = self.sample(s2_head(s1_cur, ctx), temperature, top_k, top_p)

        out1 = [s1_cur.squeeze(-1)]
        out2 = [s2_cur.squeeze(-1)]

        # ---- 自回归步: T = 1 ----
        for i in range(1, pred_len):
            stamp = full_stamp[:, lookback + i - 1 : lookback + i, :].expand(sample_count, -1, -1)
            s1_logits, ctx = decode(out1[-1].unsqueeze(-1), out2[-1].unsqueeze(-1), stamp)
            s1_cur = self.sample(s1_logits, temperature, top_k, top_p)
            s2_cur = self.sample(s2_head(s1_cur, ctx), temperature, top_k, top_p)
            out1.append(s1_cur.squeeze(-1))
            out2.append(s2_cur.squeeze(-1))

        gen_s1 = torch.stack(out1, dim=1)  # [B, P]
        gen_s2 = torch.stack(out2, dim=1)
        full_s1 = torch.cat([s1_ids, gen_s1], dim=1)
        full_s2 = torch.cat([s2_ids, gen_s2], dim=1)
        y = self.dec(full_s1, full_s2)  # [B, L+P, 6]
        return y[:, -pred_len:, :]
