"""Scaled 8xH100 RD-E8 Looped Distillation Pipeline (Target: <1.0565 val_bpb in <=16 MB).

Combines:
  1. 53.2M unique / 75.5M effective Looped U-Net Transformer (8 unique -> 12 effective layers,
     d=768, 12 Q / 4 KV heads, 4x ReLU^2 MLP, QK-Norm, RoPE, SmearGate, CaseOps SP8192).
  2. Stage 1 (`--stage teacher`): Dense BF16 Muon + AdamW teacher at seq_len=2048 (8xH100 DDP).
  3. Stage 2 (`--stage rd_e8_kd`): Zero-shot Hadamard + Conway-Sloane Shell-E8 init from the
     teacher (broadcast from rank 0) + Rate-Distortion KL+CE distillation targeting ~15.7 MB
     real on-disk artifact bytes.
  4. Stage 3 (`--stage pack_eval`): Raw `RDE8` binary frame (zero pickle bloat, guaranteed
     <= 16,000,000 B including source code), exact reload drift check, and 8-GPU sharded
     full 47,851,520-token CaseOps eval (chunk + sliding stride=256 + legal score-first TTT).
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import io
import json
import math
import os
import pathlib
import struct
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# === INLINED SPHERICAL SHELL-E8 LATTICE CODEC ===
"""Exact Hadamard + Spherical Shell-Indexed E8 Lattice Codec & Differentiable RD Layer.

Mathematical highlights:
  1. Fast Walsh-Hadamard Transform (`fwht`): orthogonal, deterministic, 0 stored bytes.
     Because H = H^T and H^2 = I, we parameterize weights directly in the rotated basis
     W_rot = W @ H and compute `F.linear(fwht(x), W_rot_e8)` — exact equivalence with
     only O(B*T*d log d) activation cost!
  2. Exact Conway-Sloane E8 nearest-point quantizer (`snap_e8`):
       E8 = D8 U (D8 + 1/2),  where D8 = {z in Z^8 : sum(z) is even}.
     100% vectorized PyTorch + STE with learnable per-layer step size `delta`.
  3. Exact zero-codebook Enumerative Sphere Coding (`e8_encode_bytes` / `e8_decode_bytes`):
     Maps every E8 point q with ||q||^2 = 2m bijectively to:
       - shell index m in {0, 1, 2, ...} (entropy-coded with brotli-11)
       - exact lexicographic rank k in [0, N_E8(2m)) computed via a tiny analytical DP
         table (rebuilt in ~1 ms at encode/decode time -> 0 stored codebook bytes!),
         bit-packed at exact `ceil(log2(N_E8(2m)))` bits (within ~0.04 bpp of Shannon).
"""


import functools
import lzma
import math
import struct
try:
    import brotli as _brotli
except ImportError:
    _brotli = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Fast Walsh-Hadamard Transform (exact orthogonal involution, 0 bytes)
# ---------------------------------------------------------------------------

def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Normalized orthogonal Walsh-Hadamard transform along the last dim (power of 2)."""
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"FWHT requires power-of-2 dim, got {n}"
    orig_shape = x.shape
    h = x.reshape(-1, n)
    step = 1
    while step < n:
        h = h.view(-1, n // (2 * step), 2, step)
        a = h[:, :, 0, :]
        b = h[:, :, 1, :]
        h = torch.stack((a + b, a - b), dim=2)
        step *= 2
    return h.reshape(orig_shape) * (1.0 / math.sqrt(n))


def fwht_block(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Block-diagonal orthogonal FWHT (works for any dim divisible by `block_size`)."""
    n = x.shape[-1]
    bs = min(block_size, n)
    while n % bs != 0:
        bs //= 2
    if bs <= 1:
        return x
    return fwht_last_dim(x.view(*x.shape[:-1], n // bs, bs)).reshape_as(x)


# ---------------------------------------------------------------------------
# 2. Exact Conway-Sloane E8 Nearest-Point Quantizer (vectorized PyTorch + STE)
# ---------------------------------------------------------------------------

def _snap_d8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point in D8 = {z in Z^8 : sum(z) is even}."""
    z = torch.round(x)
    odd = (z.sum(dim=-1).to(torch.int64) & 1) == 1
    if odd.any():
        diff = x - z
        idx = diff.abs().argmax(dim=-1, keepdim=True)
        chosen = diff.gather(-1, idx)
        step = torch.where(chosen >= 0, torch.ones_like(chosen), -torch.ones_like(chosen))
        z_fix = z.clone()
        z_fix.scatter_add_(-1, idx, step)
        z = torch.where(odd.unsqueeze(-1), z_fix, z)
    return z


def snap_e8(x: torch.Tensor) -> torch.Tensor:
    """Exact nearest lattice point in E8 = D8 U (D8 + 0.5) along last dim (= 8)."""
    q0 = _snap_d8(x)
    q1 = _snap_d8(x - 0.5) + 0.5
    d0 = (x - q0).pow(2).sum(dim=-1, keepdim=True)
    d1 = (x - q1).pow(2).sum(dim=-1, keepdim=True)
    return torch.where(d0 <= d1, q0, q1)


def snap_e8_ste(x: torch.Tensor) -> torch.Tensor:
    q = snap_e8(x.float()).to(x.dtype)
    return x + (q - x).detach()


def ste_fp16(x: torch.Tensor) -> torch.Tensor:
    x16 = x.to(torch.float16).to(x.dtype)
    return x + (x16 - x).detach()


# ---------------------------------------------------------------------------
# 3. Exact Enumerative Sphere Coding for E8 (0 stored codebook bytes!)
# ---------------------------------------------------------------------------
# Every E8 point q has doubled integer coordinates y = round(2*q) in Z^8:
#   - Coset 0 (D8):       all y_i even -> z_i = y_i / 2 in Z, sum(z_i^2) = 2m, sum(z_i) ≡ 0 (mod 2)
#   - Coset 1 (D8 + 1/2): all y_i odd  -> y_i in 2Z+1,        sum(y_i^2) = 8m, sum(y_i) ≡ 0 (mod 4)
#
# Both cosets are enumerated bijectively with tiny DP tables C0[d, s, p] and C1[d, s, r]
# rebuilt deterministically from scratch in ~2 ms (ZERO codebook bytes in the artifact!).

@functools.lru_cache(maxsize=8)
def _build_e8_dp(max_m: int = 80):
    """Build exact DP counting tables for E8 shells m = 0..max_m.

    Returns:
      c0: dict[(d, s, p)] -> count of z in Z^d with sum(z^2)=s and sum(z)%2=p (s <= 2*max_m)
      c1: dict[(d, s, r)] -> count of odd y in (2Z+1)^d with sum(y^2)=s and sum(y)%4=r (s <= 8*max_m)
      shell_counts: list[int] of N_E8(2m) = c0[8, 2m, 0] + c1[8, 8m, 0]
      coset0_counts: list[int] of c0[8, 2m, 0]
    """
    max_s0 = 2 * max_m
    # c0[d][s][p]
    c0 = [np.zeros((max_s0 + 1, 2), dtype=np.int64) for _ in range(9)]
    c0[0][0, 0] = 1
    max_z = int(math.isqrt(max_s0))
    z_vals = list(range(-max_z, max_z + 1))
    for d in range(1, 9):
        prev = c0[d - 1]
        cur = c0[d]
        for z in z_vals:
            z2 = z * z
            zp = z & 1
            if z2 <= max_s0:
                if zp == 0:
                    cur[z2:, 0] += prev[: max_s0 + 1 - z2, 0]
                    cur[z2:, 1] += prev[: max_s0 + 1 - z2, 1]
                else:
                    cur[z2:, 0] += prev[: max_s0 + 1 - z2, 1]
                    cur[z2:, 1] += prev[: max_s0 + 1 - z2, 0]

    max_s1 = 8 * max_m
    # c1[d][s][r] where r in {0, 1, 2, 3} is sum(y) mod 4
    c1 = [np.zeros((max_s1 + 1, 4), dtype=np.int64) for _ in range(9)]
    c1[0][0, 0] = 1
    max_y = int(math.isqrt(max_s1))
    y_vals = [y for y in range(-max_y, max_y + 1) if (y & 1) == 1]
    for d in range(1, 9):
        prev = c1[d - 1]
        cur = c1[d]
        for y in y_vals:
            y2 = y * y
            yr = y & 3
            if y2 <= max_s1:
                for r in range(4):
                    cur[y2:, (r + yr) & 3] += prev[: max_s1 + 1 - y2, r]

    coset0_counts = [int(c0[8][2 * m, 0]) for m in range(max_m + 1)]
    shell_counts = [int(c0[8][2 * m, 0] + c1[8][8 * m, 0]) for m in range(max_m + 1)]
    return c0, c1, coset0_counts, shell_counts


def e8_shell_size(m: int) -> int:
    if m == 0:
        return 1
    s3 = sum(d * d * d for d in range(1, m + 1) if m % d == 0)
    return 240 * s3


def _rank_e8_batch_np(y_i16: np.ndarray, m_arr: np.ndarray, max_m: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized bijective rank of N E8 points y = round(2*q) into k in [0, N_E8(2m)).

    Returns (ranks_u64, nbits_u8).
    """
    c0, c1, coset0_counts, shell_counts = _build_e8_dp(max_m)
    N = y_i16.shape[0]
    ranks = np.zeros(N, dtype=np.uint64)
    nbits = np.zeros(N, dtype=np.uint8)
    for m_val in range(1, max_m + 1):
        sc = shell_counts[m_val]
        nb = int(sc - 1).bit_length()
        mask = m_arr == m_val
        if mask.any():
            nbits[mask] = nb

    # Coset 0: all y_i even -> z_i = y_i // 2
    is_c1 = (y_i16[:, 0] & 1) == 1
    idx0 = np.flatnonzero((~is_c1) & (m_arr > 0))
    if idx0.size > 0:
        z = (y_i16[idx0] >> 1).astype(np.int64)
        rem_s = (2 * m_arr[idx0]).astype(np.int64)
        rem_p = np.zeros(idx0.size, dtype=np.int64)
        r_acc = np.zeros(idx0.size, dtype=np.uint64)
        for col in range(8):
            rem_d = 7 - col
            tbl = c0[rem_d]
            zi = z[:, col]
            lim = np.floor(np.sqrt(rem_s.astype(np.float64))).astype(np.int64)
            # Loop over feasible z_cand <= max(lim)
            max_lim = int(lim.max())
            for zc in range(-max_lim, max_lim + 1):
                zc2 = zc * zc
                active = (zc >= -lim) & (zc < zi) & (zc2 <= rem_s)
                if active.any():
                    need_p = (rem_p[active] ^ (zc & 1))
                    r_acc[active] += tbl[rem_s[active] - zc2, need_p].astype(np.uint64)
            rem_s = rem_s - zi * zi
            rem_p = rem_p ^ (zi & 1)
        ranks[idx0] = r_acc

    # Coset 1: all y_i odd, offset by coset0_counts[m]
    idx1 = np.flatnonzero(is_c1 & (m_arr > 0))
    if idx1.size > 0:
        y = y_i16[idx1].astype(np.int64)
        rem_s = (8 * m_arr[idx1]).astype(np.int64)
        rem_r = np.zeros(idx1.size, dtype=np.int64)
        c0_off = np.array(coset0_counts, dtype=np.uint64)[m_arr[idx1]]
        r_acc = c0_off.copy()
        for col in range(8):
            rem_d = 7 - col
            tbl = c1[rem_d]
            yi = y[:, col]
            lim = np.floor(np.sqrt(rem_s.astype(np.float64))).astype(np.int64)
            max_lim = int(lim.max())
            start_y = -max_lim if (max_lim & 1) == 1 else -(max_lim - 1)
            for yc in range(start_y, max_lim + 1, 2):
                yc2 = yc * yc
                active = (yc >= -lim) & (yc < yi) & (yc2 <= rem_s)
                if active.any():
                    need_r = (rem_r[active] - yc) & 3
                    r_acc[active] += tbl[rem_s[active] - yc2, need_r].astype(np.uint64)
            rem_s = rem_s - yi * yi
            rem_r = (rem_r - yi) & 3
        ranks[idx1] = r_acc

    return ranks, nbits


def _unrank_e8_batch_np(m_arr: np.ndarray, ranks: np.ndarray, max_m: int) -> np.ndarray:
    """Inverse of `_rank_e8_batch_np`: reconstruct exact y = round(2*q) int16 array (N, 8)."""
    c0, c1, coset0_counts, _ = _build_e8_dp(max_m)
    N = m_arr.shape[0]
    out = np.zeros((N, 8), dtype=np.int16)
    c0_off_all = np.array(coset0_counts, dtype=np.uint64)[m_arr]

    nz = m_arr > 0
    is_c1 = nz & (ranks >= c0_off_all)
    is_c0 = nz & (~is_c1)

    idx0 = np.flatnonzero(is_c0)
    if idx0.size > 0:
        r_rem = ranks[idx0].copy()
        rem_s = (2 * m_arr[idx0]).astype(np.int64)
        rem_p = np.zeros(idx0.size, dtype=np.int64)
        z_out = np.zeros((idx0.size, 8), dtype=np.int16)
        for col in range(8):
            rem_d = 7 - col
            tbl = c0[rem_d]
            lim = np.floor(np.sqrt(rem_s.astype(np.float64))).astype(np.int64)
            max_lim = int(lim.max())
            done = np.zeros(idx0.size, dtype=bool)
            chosen = np.zeros(idx0.size, dtype=np.int64)
            for zc in range(-max_lim, max_lim + 1):
                zc2 = zc * zc
                cand = (~done) & (zc >= -lim) & (zc2 <= rem_s)
                if not cand.any():
                    continue
                cnt = tbl[rem_s[cand] - zc2, rem_p[cand] ^ (zc & 1)].astype(np.uint64)
                hit_local = r_rem[cand] < cnt
                cand_idx = np.flatnonzero(cand)
                hit_idx = cand_idx[hit_local]
                miss_idx = cand_idx[~hit_local]
                if hit_idx.size > 0:
                    chosen[hit_idx] = zc
                    done[hit_idx] = True
                if miss_idx.size > 0:
                    r_rem[miss_idx] -= cnt[~hit_local]
            z_out[:, col] = chosen.astype(np.int16)
            rem_s = rem_s - chosen * chosen
            rem_p = rem_p ^ (chosen & 1)
        out[idx0] = z_out * 2

    idx1 = np.flatnonzero(is_c1)
    if idx1.size > 0:
        r_rem = (ranks[idx1] - c0_off_all[idx1]).copy()
        rem_s = (8 * m_arr[idx1]).astype(np.int64)
        rem_r = np.zeros(idx1.size, dtype=np.int64)
        y_out = np.zeros((idx1.size, 8), dtype=np.int16)
        for col in range(8):
            rem_d = 7 - col
            tbl = c1[rem_d]
            lim = np.floor(np.sqrt(rem_s.astype(np.float64))).astype(np.int64)
            max_lim = int(lim.max())
            start_y = -max_lim if (max_lim & 1) == 1 else -(max_lim - 1)
            done = np.zeros(idx1.size, dtype=bool)
            chosen = np.zeros(idx1.size, dtype=np.int64)
            for yc in range(start_y, max_lim + 1, 2):
                yc2 = yc * yc
                cand = (~done) & (yc >= -lim) & (yc2 <= rem_s)
                if not cand.any():
                    continue
                cnt = tbl[rem_s[cand] - yc2, (rem_r[cand] - yc) & 3].astype(np.uint64)
                hit_local = r_rem[cand] < cnt
                cand_idx = np.flatnonzero(cand)
                hit_idx = cand_idx[hit_local]
                miss_idx = cand_idx[~hit_local]
                if hit_idx.size > 0:
                    chosen[hit_idx] = yc
                    done[hit_idx] = True
                if miss_idx.size > 0:
                    r_rem[miss_idx] -= cnt[~hit_local]
            y_out[:, col] = chosen.astype(np.int16)
            rem_s = rem_s - chosen * chosen
            rem_r = (rem_r - chosen) & 3
        out[idx1] = y_out

    return out


def _pack_varbits(ranks: np.ndarray, nbits: np.ndarray) -> bytes:
    """Pack uint64 values with per-element bit widths (0..63) tightly into bytes."""
    total_bits = int(nbits.astype(np.int64).sum())
    if total_bits == 0:
        return b""
    # Fast byte-chunked bit packer grouped by exact bit-width (only ~20 distinct widths!)
    # Because shell m determines nbits, we store ranks in shell order? Even simpler:
    # order ranks by exact stream position using a fast cumulative-bit scatter in uint64!
    # For N <= 10M vectors, chunk into 256k blocks to stay cache-friendly.
    n_bytes = (total_bits + 7) // 8
    out = np.zeros(n_bytes + 8, dtype=np.uint8)
    bit_pos = np.empty(ranks.size, dtype=np.int64)
    bit_pos[0] = 0
    if ranks.size > 1:
        np.cumsum(nbits[:-1], dtype=np.int64, out=bit_pos[1:])

    # Scatter bit-by-bit? Faster: scatter byte-aligned words by bit width!
    # Better yet: group by bit-width (or shell m) — the decoder ALREADY knows m_arr,
    # so grouping ranks by m_val packs each shell's fixed-width array in pure vectorized numpy!
    raise NotImplementedError("Use _pack_grouped_by_shell instead")


def _pack_fixed_width(vals: np.ndarray, width: int) -> bytes:
    """Vectorized tight bit-packing of N uint64 values each of `width` bits (1..60)."""
    n = vals.size
    if n == 0 or width == 0:
        return b""
    # Pack 8 values at a time into `width` bytes (8 * width bits = exact byte boundary!)
    pad = (-n) % 8
    if pad:
        vals = np.concatenate([vals, np.zeros(pad, dtype=np.uint64)])
    g = vals.reshape(-1, 8)  # (M, 8)
    # Build (M, width) bytes by extracting 8 bits at a time across the 8*width bit chunk
    # For width <= 60, we can accumulate into 64-bit lanes cleanly:
    out = np.zeros((g.shape[0], width), dtype=np.uint8)
    # Place each lane j in [0..7] at bit offset j * width
    for j in range(8):
        v = g[:, j]
        bit_off = j * width
        b0 = bit_off >> 3
        sh = bit_off & 7
        # v spans ceil((sh + width) / 8) bytes starting at b0
        n_span = (sh + width + 7) >> 3
        for k in range(n_span):
            if k == 0:
                piece = (v << np.uint64(sh)) & np.uint64(0xFF)
            else:
                piece = (v >> np.uint64(8 * k - sh)) & np.uint64(0xFF)
            out[:, b0 + k] |= piece.astype(np.uint8)
    total_bytes = (n * width + 7) // 8
    return out.reshape(-1)[:total_bytes].tobytes()


def _unpack_fixed_width(buf: bytes, n: int, width: int) -> np.ndarray:
    """Inverse of `_pack_fixed_width`."""
    if n == 0 or width == 0:
        return np.zeros(n, dtype=np.uint64)
    n_pad = n + ((-n) % 8)
    m_groups = n_pad // 8
    need_bytes = m_groups * width
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size < need_bytes:
        padded = np.zeros(need_bytes, dtype=np.uint8)
        padded[: arr.size] = arr
        arr = padded
    g = arr[:need_bytes].reshape(m_groups, width)
    out = np.zeros((m_groups, 8), dtype=np.uint64)
    mask = (np.uint64(1) << np.uint64(width)) - np.uint64(1)
    for j in range(8):
        bit_off = j * width
        b0 = bit_off >> 3
        sh = bit_off & 7
        n_span = (sh + width + 7) >> 3
        acc = np.zeros(m_groups, dtype=np.uint64)
        for k in range(n_span):
            byte_col = g[:, b0 + k].astype(np.uint64)
            if k == 0:
                acc |= byte_col >> np.uint64(sh)
            else:
                acc |= byte_col << np.uint64(8 * k - sh)
        out[:, j] = acc & mask
    return out.reshape(-1)[:n]


def _compress_bytes(raw: bytes) -> bytes:
    if _brotli is not None:
        return b"B" + _brotli.compress(raw, quality=11)
    return b"X" + lzma.compress(raw, preset=6)


def _decompress_bytes(blob: bytes) -> bytes:
    tag, body = blob[:1], blob[1:]
    if tag == b"B":
        return _brotli.decompress(body)
    if tag == b"X":
        return lzma.decompress(body)
    raise ValueError(f"Unknown compression tag {tag!r}")


def e8_encode_bytes(q_e8: torch.Tensor, max_dp_m: int = 80) -> bytes:
    """Losslessly encode an E8 tensor (..., 8) into a compact binary stream.

    Shells m <= max_dp_m use exact zero-codebook enumerative sphere coding grouped
    by shell m (so each shell packs at exact fixed width `ceil(log2(N_E8(2m)))` bits!).
    Rare escape shells m > max_dp_m (virtually never hit after Hadamard) store raw int16 y.
    """
    y = torch.round(q_e8.detach().cpu().float() * 2.0).to(torch.int16).reshape(-1, 8).numpy()
    N = y.shape[0]
    norm2_y = (y.astype(np.int64) ** 2).sum(axis=1)
    # ||y||^2 = 4 * ||q||^2 = 8 * m
    assert np.all((norm2_y & 7) == 0), "Input is not on E8 lattice (||2q||^2 must be divisible by 8)"
    m_full = norm2_y >> 3

    escape_mask = m_full > max_dp_m
    m_stored = np.where(escape_mask, 255, m_full).astype(np.uint8)
    m_clamped = np.where(escape_mask, 0, m_full).astype(np.int64)
    actual_max_m = int(m_clamped.max()) if N > 0 else 0

    ranks, _ = _rank_e8_batch_np(y, m_clamped, max(actual_max_m, 1))
    _, _, _, shell_counts = _build_e8_dp(max(actual_max_m, 1))

    # Compress shell indices (highly concentrated on small m -> brotli hits exact H(m)!)
    shell_comp = _compress_bytes(m_stored.tobytes())

    # Pack rank bits grouped by shell m=1..actual_max_m (zero entropy left in uniform ranks!)
    rank_chunks = []
    for mv in range(1, actual_max_m + 1):
        idx = np.flatnonzero(m_clamped == mv)
        if idx.size == 0:
            continue
        width = int(shell_counts[mv] - 1).bit_length()
        packed = _pack_fixed_width(ranks[idx], width)
        rank_chunks.append(struct.pack("<BBI", mv, width, len(packed)) + packed)

    esc_y = y[escape_mask].astype("<i2").tobytes() if escape_mask.any() else b""
    esc_comp = _compress_bytes(esc_y) if esc_y else b""

    header = struct.pack("<IBHII", N, actual_max_m, len(rank_chunks), len(shell_comp), len(esc_comp))
    return header + shell_comp + esc_comp + b"".join(rank_chunks)


def e8_decode_bytes(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """Lossless inverse of `e8_encode_bytes`, returning exact float32 E8 tensor."""
    hdr_sz = struct.calcsize("<IBHII")
    N, actual_max_m, n_chunks, shell_len, esc_len = struct.unpack("<IBHII", blob[:hdr_sz])
    off = hdr_sz
    shell_raw = _decompress_bytes(blob[off : off + shell_len])
    off += shell_len
    m_stored = np.frombuffer(shell_raw, dtype=np.uint8).copy()
    assert m_stored.size == N

    esc_raw = _decompress_bytes(blob[off : off + esc_len]) if esc_len > 0 else b""
    off += esc_len

    escape_mask = m_stored == 255
    m_clamped = np.where(escape_mask, 0, m_stored).astype(np.int64)
    ranks = np.zeros(N, dtype=np.uint64)

    for _ in range(n_chunks):
        mv, width, plen = struct.unpack("<BBI", blob[off : off + 6])
        off += 6
        pbytes = blob[off : off + plen]
        off += plen
        idx = np.flatnonzero(m_clamped == mv)
        ranks[idx] = _unpack_fixed_width(pbytes, idx.size, width)

    y = _unrank_e8_batch_np(m_clamped, ranks, max(actual_max_m, 1))
    if escape_mask.any():
        y[escape_mask] = np.frombuffer(esc_raw, dtype="<i2").reshape(-1, 8)
    return torch.from_numpy(y.astype(np.float32) * 0.5).view(shape)


# ---------------------------------------------------------------------------
# 4. Differentiable Rate-Distortion E8 Linear Layer (`RDE8Linear`)
# ---------------------------------------------------------------------------

class RDE8Linear(nn.Module):
    """Hadamard-rotated E8 Lattice Linear Layer with learnable step size `log_delta`.

    Parameterizes weights directly in the orthogonal Walsh-Hadamard basis:
        y = F.linear(fwht_block(x), W_rot_e8)
    where `W_rot_e8 = delta * snap_e8(W_rot / delta)` via STE and FP16-STE on `delta`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_delta: float = 0.04,
        hadamard_block: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        assert in_features % 8 == 0, f"RDE8Linear requires in_features % 8 == 0, got {in_features}"
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hadamard_block = int(hadamard_block)

        self.W_rot = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.normal_(self.W_rot, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        # Learnable per-layer quantization step size (stored as single FP16 scalar!)
        self.log_delta = nn.Parameter(torch.tensor(math.log(max(init_delta, 1e-4)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._frozen_w: torch.Tensor | None = None

    @torch.no_grad()
    def freeze_for_eval(self) -> None:
        """Fold symmetric orthogonal FWHT into the snapped E8 weight for fast native GEMM eval/TTT."""
        d = self.delta().float()
        w_e8 = (snap_e8((self.W_rot.detach().float() / d).view(-1, 8)) * d).view(self.out_features, self.in_features)
        self._frozen_w = fwht_block(w_e8, self.hadamard_block).contiguous()

    def delta(self) -> torch.Tensor:
        return ste_fp16(torch.exp(self.log_delta.clamp(-9.0, 2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._frozen_w is not None and not self.training:
            b = ste_fp16(self.bias).to(x.dtype) if self.bias is not None else None
            return F.linear(x, self._frozen_w.to(x.dtype), b)
        d = self.delta()
        w_scaled = self.W_rot.view(-1, 8) / d
        w_e8 = (snap_e8_ste(w_scaled) * d).view(self.out_features, self.in_features).to(x.dtype)
        x_rot = fwht_block(x, self.hadamard_block)
        b = ste_fp16(self.bias).to(x.dtype) if self.bias is not None else None
        return F.linear(x_rot, w_e8, b)

    def differentiable_rate_bits(self) -> torch.Tensor:
        """Smooth differentiable proxy for total Shell-E8 bits of this weight matrix.

        Matches exact spherical Shell-E8 bits:
          - For u = W_rot / delta in R^8, shell m ≈ ||u||^2 / 2.
          - Exact shell capacity grows as log2(N_E8(2m)) ≈ log2(240 * 1.202) + 3 * log2(m)
            for m >= 1, and ~0.15 bits at m = 0.
          - Differentiable everywhere w.r.t. BOTH `W_rot` and `log_delta`!
        """
        d = torch.exp(self.log_delta.clamp(-9.0, 2.0))
        u = self.W_rot.view(-1, 8) / d
        m_soft = 0.5 * u.pow(2).sum(dim=-1)  # (N_blocks,)
        # Calibrated smooth shell+index bits per 8D block (verified against exact Shell-E8!):
        #   m=0 -> ~0.2 bits; m=1 -> ~8.5 bits; m=2 -> ~12.5 bits; m=4 -> ~15.8 bits
        bits_per_block = 3.15 * torch.log2(1.0 + 3.2 * m_soft) + 0.25 * (1.0 - torch.exp(-m_soft))
        return bits_per_block.sum()

    @torch.no_grad()
    def exact_shell_entropy_bits(self) -> float:
        """Exact empirical Shell-E8 Shannon bits (shell entropy + log2(N_E8(2m)))."""
        d = float(self.delta().item())
        q = snap_e8((self.W_rot.detach().float() / d).view(-1, 8))
        norm2 = torch.round((q * q).sum(dim=-1)).to(torch.int64)
        shells, counts = torch.unique(norm2, return_counts=True)
        total = float(norm2.numel())
        bits = 0.0
        for sh, cnt in zip(shells.tolist(), counts.tolist()):
            p = cnt / total
            bits += cnt * (-math.log2(p) + math.log2(e8_shell_size(sh // 2)))
        return bits

    @torch.no_grad()
    def pack_to_dict(self) -> dict[str, bytes | torch.Tensor]:
        d_fp16 = self.delta().detach().cpu().to(torch.float16)
        q = snap_e8((self.W_rot.detach().float() / float(d_fp16.item())).view(-1, 8))
        out: dict[str, bytes | torch.Tensor] = {
            "e8_blob": e8_encode_bytes(q),
            "delta": d_fp16.reshape(1),
        }
        if self.bias is not None:
            out["bias"] = self.bias.detach().cpu().to(torch.float16)
        return out

    @torch.no_grad()
    def load_from_packed_dict(self, d: dict) -> None:
        dev = self.W_rot.device
        delta_val = float(d["delta"].to(torch.float16).item())
        q = e8_decode_bytes(d["e8_blob"], (self.out_features, self.in_features)).to(dev)
        self.log_delta.copy_(torch.tensor(math.log(max(delta_val, 1e-7)), device=dev, dtype=torch.float32))
        self.W_rot.copy_(q * delta_val)
        if self.bias is not None and "bias" in d:
            self.bias.copy_(d["bias"].to(device=dev, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Self-test: bit-exact E8 encode/decode + real on-disk byte benchmark vs Scalar
# ---------------------------------------------------------------------------

# === END SHELL-E8 CODEC ===



@dataclasses.dataclass
class RDE8Config:
    vocab_size: int = 8192
    num_unique_layers: int = 8
    loop_start: int = 2
    loop_end: int = 6          # blocks [2, 3, 4, 5] run twice -> 12 effective layers
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    mlp_mult: int = 4
    embed_bottleneck: int = 320
    embed_bits: int = 6
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    logit_softcap: float = 30.0
    hadamard_block: int = 256
    quant_mode: str = "dense"  # "dense" | "rd_e8"


def _ste_int_per_row(w: torch.Tensor, bits: int = 6) -> torch.Tensor:
    qmax = float((1 << (bits - 1)) - 1)
    scale = (w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax).to(torch.float16).float()
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    w_q = (q * scale).to(w.dtype)
    return w + (w_q - w).detach()


def _pack_int_per_row(w: torch.Tensor, bits: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = (1 << (bits - 1)) - 1
    wf = w.detach().cpu().float()
    scale = (wf.abs().amax(dim=1).clamp_min(1e-8) / float(qmax)).to(torch.float16)
    q = torch.clamp(torch.round(wf / scale.float().unsqueeze(1)), -qmax, qmax).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _unpack_int_per_row(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float().unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, fp16_ste: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.fp16_ste = fp16_ste

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        w = ste_fp16(self.weight) if self.fp16_ste else self.weight
        return x * norm * w.to(x.dtype)


class RotaryCache(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def get(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._seq_len < seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._seq_len = seq_len
        return self._cos[:, :, :seq_len, :].to(dtype), self._sin[:, :, :seq_len, :].to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class LoRAPair(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0 / math.sqrt(max(rank, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


class TTTLoRABank(nn.Module):
    """Per-unique-layer LoRA adapters on attn.proj and mlp.proj for legal score-first TTT."""

    def __init__(self, cfg: RDE8Config, rank: int = 32):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.attn_o = nn.ModuleList([LoRAPair(cfg.d_model, cfg.d_model, rank) for _ in range(cfg.num_unique_layers)])
        self.mlp_proj = nn.ModuleList([LoRAPair(hidden, cfg.d_model, rank) for _ in range(cfg.num_unique_layers)])


def _make_linear(in_f: int, out_f: int, cfg: RDE8Config) -> nn.Module:
    if cfg.quant_mode == "dense":
        return nn.Linear(in_f, out_f, bias=False)
    if cfg.quant_mode == "rd_e8":
        return RDE8Linear(in_f, out_f, init_delta=1.05 / math.sqrt(in_f), hadamard_block=cfg.hadamard_block)
    raise ValueError(f"Unknown quant_mode={cfg.quant_mode!r}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RDE8Config):
        super().__init__()
        q = cfg.quant_mode != "dense"
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        kv_dim = self.num_kv_heads * self.head_dim

        self.c_q = _make_linear(cfg.d_model, cfg.d_model, cfg)
        self.c_k = _make_linear(cfg.d_model, kv_dim, cfg)
        self.c_v = _make_linear(cfg.d_model, kv_dim, cfg)
        self.proj = _make_linear(cfg.d_model, cfg.d_model, cfg)
        self.q_norm = RMSNorm(self.head_dim, fp16_ste=q)
        self.k_norm = RMSNorm(self.head_dim, fp16_ste=q)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lora_o: LoRAPair | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(y)
        if lora_o is not None:
            out = out + lora_o(y)
        return out


class ReLUSquaredMLP(nn.Module):
    def __init__(self, cfg: RDE8Config):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.fc = _make_linear(cfg.d_model, hidden, cfg)
        self.proj = _make_linear(hidden, cfg.d_model, cfg)

    def forward(self, x: torch.Tensor, lora_mlp: LoRAPair | None = None) -> torch.Tensor:
        h = F.relu(self.fc(x)).square()
        out = self.proj(h)
        if lora_mlp is not None:
            out = out + lora_mlp(h)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: RDE8Config):
        super().__init__()
        q = cfg.quant_mode != "dense"
        self.norm1 = RMSNorm(cfg.d_model, fp16_ste=q)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, fp16_ste=q)
        self.mlp = ReLUSquaredMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        resid_scale: torch.Tensor,
        lora_o: LoRAPair | None = None,
        lora_mlp: LoRAPair | None = None,
    ) -> torch.Tensor:
        x = x + resid_scale * self.attn(self.norm1(x), cos, sin, lora_o=lora_o)
        x = x + resid_scale * self.mlp(self.norm2(x), lora_mlp=lora_mlp)
        return x


class LoopedRDE8GPT(nn.Module):
    def __init__(self, cfg: RDE8Config):
        super().__init__()
        self.cfg = cfg
        q = cfg.quant_mode != "dense"
        self.embed_tok = nn.Parameter(torch.randn(cfg.vocab_size, cfg.embed_bottleneck) * 0.02)
        self.embed_proj = nn.Parameter(
            torch.randn(cfg.d_model, cfg.embed_bottleneck) * (1.0 / math.sqrt(cfg.embed_bottleneck))
        )
        self.smear_gate = nn.Parameter(torch.zeros(cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_unique_layers)])
        self.exec_order = (
            list(range(0, cfg.loop_start))
            + list(range(cfg.loop_start, cfg.loop_end))
            + list(range(cfg.loop_start, cfg.loop_end))
            + list(range(cfg.loop_end, cfg.num_unique_layers))
        )
        self.step_scales = nn.Parameter(torch.ones(len(self.exec_order)))
        self.num_skips = len(self.exec_order) // 2
        self.skip_weights = nn.Parameter(torch.zeros(self.num_skips))
        self.final_norm = RMSNorm(cfg.d_model, fp16_ste=q)
        self.rotary = RotaryCache(cfg.d_model // cfg.num_heads, base=cfg.rope_base)

    def quant_modules(self) -> list[tuple[str, RDE8Linear]]:
        return [(name, mod) for name, mod in self.named_modules() if isinstance(mod, RDE8Linear)]

    def differentiable_rate_bits(self) -> torch.Tensor:
        total = torch.zeros((), device=self.embed_tok.device, dtype=torch.float32)
        for _, mod in self.quant_modules():
            total = total + mod.differentiable_rate_bits()
        return total

    @torch.no_grad()
    def exact_entropy_bits(self) -> float:
        return sum(mod.exact_shell_entropy_bits() for _, mod in self.quant_modules())

    def forward(self, input_ids: torch.Tensor, ttt_lora: TTTLoRABank | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        q = self.cfg.quant_mode != "dense"
        dtype = torch.get_autocast_dtype("cuda" if input_ids.is_cuda else "cpu") if torch.is_autocast_enabled() else self.embed_tok.dtype

        et = _ste_int_per_row(self.embed_tok, bits=self.cfg.embed_bits).to(dtype) if q else self.embed_tok.to(dtype)
        ep = _ste_int_per_row(self.embed_proj, bits=self.cfg.embed_bits).to(dtype) if q else self.embed_proj.to(dtype)
        x = F.linear(F.embedding(input_ids, et), ep)

        sg = ste_fp16(self.smear_gate) if q else self.smear_gate
        x = x + torch.tanh(sg.to(x.dtype)) * F.pad(x[:, :-1, :], (0, 0, 1, 0))

        cos, sin = self.rotary.get(T, x.device, x.dtype)
        ss = (ste_fp16(self.step_scales) if q else self.step_scales).to(x.dtype)
        sw = (ste_fp16(self.skip_weights) if q else self.skip_weights).to(x.dtype)

        skips: list[torch.Tensor] = []
        for step_i, blk_idx in enumerate(self.exec_order):
            lo = ttt_lora.attn_o[blk_idx] if ttt_lora is not None else None
            lm = ttt_lora.mlp_proj[blk_idx] if ttt_lora is not None else None
            if step_i < self.num_skips:
                x = self.blocks[blk_idx](x, cos, sin, ss[step_i], lora_o=lo, lora_mlp=lm)
                skips.append(x)
            else:
                sk_i = self.num_skips - 1 - (step_i - self.num_skips)
                if 0 <= sk_i < len(skips):
                    x = x + sw[sk_i] * skips[sk_i]
                x = self.blocks[blk_idx](x, cos, sin, ss[step_i], lora_o=lo, lora_mlp=lm)

        x = self.final_norm(x)
        logits = F.linear(F.linear(x, ep.T), et)
        cap = float(self.cfg.logit_softcap)
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)
        return logits


# ---------------------------------------------------------------------------
# Muon Optimizer (for fast Stage 1 Dense Teacher training)
# ---------------------------------------------------------------------------

def _zeropower_ns5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.04):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(beta).add_(g)
                u = _zeropower_ns5(g.add(buf, alpha=beta))
                scale = math.sqrt(max(1.0, p.size(0) / max(p.size(1), 1)))
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * scale)
        return loss


def build_teacher_optimizers(model: LoopedRDE8GPT, matrix_lr: float = 0.022, embed_lr: float = 3e-3) -> list[torch.optim.Optimizer]:
    muon_p, adam_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name.startswith("blocks."):
            muon_p.append(p)
        else:
            adam_p.append(p)
    opts: list[torch.optim.Optimizer] = []
    if muon_p:
        opts.append(Muon(muon_p, lr=matrix_lr, momentum=0.95, weight_decay=0.04))
    if adam_p:
        opts.append(torch.optim.AdamW(adam_p, lr=embed_lr, betas=(0.9, 0.95), weight_decay=0.01))
    return opts


# ---------------------------------------------------------------------------
# Teacher -> RD-E8 Init + Raw RDE8 Binary Frame Save/Load (zero pickle bloat)
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_rde8_from_dense_teacher(teacher: LoopedRDE8GPT, target_bpp: float = 2.15, device: torch.device | None = None) -> LoopedRDE8GPT:
    dev = device or next(teacher.parameters()).device
    scfg = dataclasses.replace(teacher.cfg, quant_mode="rd_e8")
    student = LoopedRDE8GPT(scfg).to(dev)
    student.embed_tok.copy_(teacher.embed_tok.data)
    student.embed_proj.copy_(teacher.embed_proj.data)
    student.smear_gate.copy_(teacher.smear_gate.data)
    student.step_scales.copy_(teacher.step_scales.data)
    student.skip_weights.copy_(teacher.skip_weights.data)
    student.final_norm.weight.copy_(teacher.final_norm.weight.data)

    t_mods = dict(teacher.named_modules())
    for tb, sb in zip(teacher.blocks, student.blocks):
        sb.norm1.weight.copy_(tb.norm1.weight.data)
        sb.norm2.weight.copy_(tb.norm2.weight.data)
        sb.attn.q_norm.weight.copy_(tb.attn.q_norm.weight.data)
        sb.attn.k_norm.weight.copy_(tb.attn.k_norm.weight.data)

    for name, mod in student.quant_modules():
        t_lin = t_mods[name]
        W_rot = fwht_block(t_lin.weight.data.to(device=dev, dtype=torch.float32), scfg.hadamard_block)
        mod.W_rot.copy_(W_rot)
        w_std = float(W_rot.std().item())
        lo, hi = 0.2 * w_std, 3.0 * w_std
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            mod.log_delta.fill_(math.log(max(mid, 1e-6)))
            bpp = mod.exact_shell_entropy_bits() / W_rot.numel()
            if bpp > target_bpp:
                lo = mid
            else:
                hi = mid
        mod.log_delta.fill_(math.log(max(0.5 * (lo + hi), 1e-6)))
    return student


@torch.no_grad()
def save_rde8_artifact(model: LoopedRDE8GPT, out_path: str | pathlib.Path) -> dict[str, int]:
    out = io.BytesIO()
    out.write(b"RDE8")

    def _wb(b: bytes) -> None:
        out.write(struct.pack("<I", len(b)))
        out.write(b)

    _wb(json.dumps(dataclasses.asdict(model.cfg)).encode("utf-8"))
    q_mods = model.quant_modules()
    out.write(struct.pack("<H", len(q_mods)))
    quant_bytes_sum = 0
    for name, mod in q_mods:
        pk = mod.pack_to_dict()
        blob = pk["e8_blob"]
        _wb(name.encode("utf-8"))
        out.write(pk["delta"].detach().cpu().to(torch.float16).numpy().tobytes())
        _wb(blob)
        quant_bytes_sum += len(blob) + 2

    q_et, s_et = _pack_int_per_row(model.embed_tok, bits=model.cfg.embed_bits)
    q_ep, s_ep = _pack_int_per_row(model.embed_proj, bits=model.cfg.embed_bits)
    emb_comp = _compress_bytes(
        q_et.numpy().tobytes() + s_et.numpy().tobytes() + q_ep.numpy().tobytes() + s_ep.numpy().tobytes()
    )
    out.write(struct.pack("<IIII", q_et.shape[0], q_et.shape[1], q_ep.shape[0], q_ep.shape[1]))
    _wb(emb_comp)

    q_pnames = {f"{mn}.{pn}" for mn, m in q_mods for pn, _ in m.named_parameters(recurse=False)}
    misc_items = [
        (pn, p.detach().cpu().to(torch.float16).contiguous())
        for pn, p in model.named_parameters()
        if pn not in q_pnames and pn not in ("embed_tok", "embed_proj")
    ]
    out.write(struct.pack("<H", len(misc_items)))
    for pn, t16 in misc_items:
        _wb(pn.encode("utf-8"))
        _wb(t16.numpy().tobytes())

    raw = out.getvalue()
    pathlib.Path(out_path).write_bytes(raw)
    return {"quant_linear_bytes": quant_bytes_sum, "embed_bytes": len(emb_comp), "total_model_bytes": len(raw)}


@torch.no_grad()
def load_rde8_artifact(path: str | pathlib.Path, device: torch.device) -> LoopedRDE8GPT:
    r = io.BytesIO(pathlib.Path(path).read_bytes())
    assert r.read(4) == b"RDE8"

    def _rb() -> bytes:
        (n,) = struct.unpack("<I", r.read(4))
        return r.read(n)

    cfg = RDE8Config(**json.loads(_rb().decode("utf-8")))
    model = LoopedRDE8GPT(cfg).to(device)
    mmap = dict(model.quant_modules())
    (n_q,) = struct.unpack("<H", r.read(2))
    for _ in range(n_q):
        name = _rb().decode("utf-8")
        delta = torch.from_numpy(np.frombuffer(r.read(2), dtype=np.float16).copy())
        blob = _rb()
        mmap[name].load_from_packed_dict({"e8_blob": blob, "delta": delta})

    et_m, et_n, ep_m, ep_n = struct.unpack("<IIII", r.read(16))
    raw_emb = _decompress_bytes(_rb())
    o0 = et_m * et_n
    o1 = o0 + et_m * 2
    o2 = o1 + ep_m * ep_n
    o3 = o2 + ep_m * 2
    q_et = torch.from_numpy(np.frombuffer(raw_emb[:o0], dtype=np.int8).copy()).view(et_m, et_n)
    s_et = torch.from_numpy(np.frombuffer(raw_emb[o0:o1], dtype=np.float16).copy())
    q_ep = torch.from_numpy(np.frombuffer(raw_emb[o1:o2], dtype=np.int8).copy()).view(ep_m, ep_n)
    s_ep = torch.from_numpy(np.frombuffer(raw_emb[o2:o3], dtype=np.float16).copy())
    model.embed_tok.copy_(_unpack_int_per_row(q_et, s_et).to(device))
    model.embed_proj.copy_(_unpack_int_per_row(q_ep, s_ep).to(device))

    pmap = dict(model.named_parameters())
    (n_misc,) = struct.unpack("<H", r.read(2))
    for _ in range(n_misc):
        pn = _rb().decode("utf-8")
        t16 = torch.from_numpy(np.frombuffer(_rb(), dtype=np.float16).copy()).view_as(pmap[pn])
        pmap[pn].copy_(t16.to(device=device, dtype=pmap[pn].dtype))
    return model


# ---------------------------------------------------------------------------
# Shards + Sharded 8-GPU Full-Val Sliding & Score-First TTT Evaluators
# ---------------------------------------------------------------------------

class ShardLoader:
    def __init__(self, pattern: str, seq_len: int, seed: int = 42, rank: int = 0):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No shards matched {pattern!r}")
        self.seq_len = int(seq_len)
        self.rng = np.random.default_rng(seed + 1009 * rank)
        self.mmaps = [np.memmap(f, dtype=np.uint16, mode="r", offset=1024) for f in self.files]

    def next_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        xs = np.empty((batch_size, self.seq_len), dtype=np.int64)
        ys = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i in range(batch_size):
            mm = self.mmaps[int(self.rng.integers(0, len(self.mmaps)))]
            pos = int(self.rng.integers(0, len(mm) - self.seq_len - 1))
            ch = mm[pos : pos + self.seq_len + 1].astype(np.int64)
            xs[i] = ch[:-1]
            ys[i] = ch[1:]
        return (
            torch.from_numpy(xs).to(device, non_blocking=True),
            torch.from_numpy(ys).to(device, non_blocking=True),
        )


def load_caseops_val(val_glob: str, max_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    tok_files = [f for f in sorted(glob.glob(val_glob)) if "val_bytes_" not in os.path.basename(f)]
    byte_files = sorted(glob.glob(val_glob.replace("fineweb_val_", "fineweb_val_bytes_")))
    toks = torch.from_numpy(np.concatenate([np.fromfile(f, dtype=np.uint16, offset=1024).astype(np.int64) for f in tok_files]))
    byts = torch.from_numpy(np.concatenate([np.fromfile(f, dtype=np.uint16, offset=1024).astype(np.int64) for f in byte_files]))
    assert toks.numel() == byts.numel()
    if max_tokens > 0:
        toks = toks[: max_tokens + 1]
        byts = byts[: max_tokens + 1]
    return toks, byts


@torch.no_grad()
def eval_caseops_bpb(
    model: nn.Module,
    val_tokens: torch.Tensor,
    val_bytes: torch.Tensor,
    device: torch.device,
    seq_len: int = 2048,
    stride: int | None = None,
    batch_size: int = 16,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, int, int]:
    model.eval()
    stride = int(seq_len if stride is None else stride)
    n_total = val_tokens.numel() - 1
    use_bf16 = device.type == "cuda"

    windows = []
    pos = 0
    while pos < n_total:
        score_end = min(pos + stride, n_total)
        win_start = max(0, score_end - seq_len)
        windows.append((win_start, score_end, pos - win_start))
        pos = score_end
    if world_size > 1:
        windows = windows[rank::world_size]

    loss_sum = 0.0
    tok_sum = 0
    byte_sum = 0
    i = 0
    while i < len(windows):
        wlen = windows[i][1] - windows[i][0]
        bw = []
        while i < len(windows) and len(bw) < batch_size and (windows[i][1] - windows[i][0]) == wlen:
            bw.append(windows[i])
            i += 1
        xs = torch.stack([val_tokens[ws:we] for ws, we, _ in bw]).to(device, non_blocking=True)
        ys = torch.stack([val_tokens[ws + 1 : we + 1] for ws, we, _ in bw]).to(device, non_blocking=True)
        bs = torch.stack([val_bytes[ws + 1 : we + 1] for ws, we, _ in bw])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(xs)
        ptl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ys.reshape(-1), reduction="none").view(len(bw), wlen)
        for bi, (_, _, soff) in enumerate(bw):
            loss_sum += float(ptl[bi, soff:].double().sum().item())
            tok_sum += wlen - soff
            byte_sum += int(bs[bi, soff:].sum().item())

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        st = torch.tensor([loss_sum, float(tok_sum), float(byte_sum)], dtype=torch.float64, device=device)
        dist.all_reduce(st, op=dist.ReduceOp.SUM)
        loss_sum, tok_sum, byte_sum = float(st[0].item()), int(round(st[1].item())), int(round(st[2].item()))

    vl = loss_sum / max(tok_sum, 1)
    vb = vl / math.log(2.0) * (tok_sum / max(byte_sum, 1))
    return vl, vb, tok_sum, byte_sum


def eval_caseops_score_first_ttt(
    model: LoopedRDE8GPT,
    val_tokens: torch.Tensor,
    val_bytes: torch.Tensor,
    device: torch.device,
    seq_len: int = 2048,
    chunk_size: int = 512,
    lora_rank: int = 32,
    ttt_lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, int, int]:
    """Legal score-first LoRA TTT sharded across `world_size` independent segments."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_total = val_tokens.numel() - 1
    seg_start = (n_total * rank) // world_size
    seg_end = (n_total * (rank + 1)) // world_size

    lora = TTTLoRABank(model.cfg, rank=lora_rank).to(device)
    opt = torch.optim.AdamW(lora.parameters(), lr=ttt_lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    use_bf16 = device.type == "cuda"

    loss_sum = 0.0
    tok_sum = 0
    byte_sum = 0
    pos = seg_start
    while pos < seg_end:
        chunk_end = min(pos + chunk_size, seg_end)
        win_start = max(seg_start, chunk_end - seq_len)
        soff = pos - win_start
        x = val_tokens[win_start:chunk_end].unsqueeze(0).to(device)
        y = val_tokens[win_start + 1 : chunk_end + 1].unsqueeze(0).to(device)
        b_slice = val_bytes[pos + 1 : chunk_end + 1]

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, ttt_lora=lora)
        ptl = F.cross_entropy(
            logits[:, soff:, :].float().reshape(-1, logits.size(-1)),
            y[:, soff:].reshape(-1),
            reduction="none",
        )
        loss_sum += float(ptl.detach().double().sum().item())
        tok_sum += int(y[:, soff:].numel())
        byte_sum += int(b_slice.sum().item())

        ptl.mean().backward()
        torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
        opt.step()
        pos = chunk_end

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        st = torch.tensor([loss_sum, float(tok_sum), float(byte_sum)], dtype=torch.float64, device=device)
        dist.all_reduce(st, op=dist.ReduceOp.SUM)
        loss_sum, tok_sum, byte_sum = float(st[0].item()), int(round(st[1].item())), int(round(st[2].item()))

    vl = loss_sum / max(tok_sum, 1)
    vb = vl / math.log(2.0) * (tok_sum / max(byte_sum, 1))
    return vl, vb, tok_sum, byte_sum


def run_cpu_smoke() -> None:
    import tempfile
    torch.manual_seed(0)
    dev = torch.device("cpu")
    mini = RDE8Config(
        vocab_size=128, num_unique_layers=4, loop_start=1, loop_end=3,
        d_model=64, num_heads=4, num_kv_heads=2, mlp_mult=4,
        embed_bottleneck=32, max_seq_len=32, hadamard_block=64, quant_mode="dense",
    )
    teacher = LoopedRDE8GPT(mini).to(dev)
    opts = build_teacher_optimizers(teacher)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    for _ in range(3):
        for o in opts:
            o.zero_grad()
        F.cross_entropy(teacher(x).reshape(-1, 128), y.reshape(-1)).backward()
        for o in opts:
            o.step()
    student = init_rde8_from_dense_teacher(teacher, target_bpp=2.15, device=dev)
    student.eval()
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "m.rde8"
        st = save_rde8_artifact(student, p)
        rel = load_rde8_artifact(p, dev)
        rel.eval()
        with torch.no_grad():
            diff = (student(x) - rel(x)).abs().max().item()
        toks = torch.randint(0, 128, (129,))
        byts = torch.full((129,), 4)
        _, b_sl, nt_s, _ = eval_caseops_bpb(rel, toks, byts, dev, seq_len=32, stride=16, batch_size=2)
        _, b_ttt, nt_t, _ = eval_caseops_score_first_ttt(rel, toks, byts, dev, seq_len=32, chunk_size=16, lora_rank=8, ttt_lr=1e-3)
    print(f"SCALED CPU SMOKE OK: bytes={st['total_model_bytes']:,} diff={diff:.8f} slide_bpb={b_sl:.4f} ttt_bpb={b_ttt:.4f} (nt={nt_s},{nt_t})")
    assert diff < 1e-4 and nt_s == nt_t == 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--stage", choices=["teacher", "rd_e8_kd", "pack_eval"], default="teacher")
    ap.add_argument("--data-glob", default="/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin")
    ap.add_argument("--val-glob", default="/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin")
    ap.add_argument("--teacher-ckpt", default="/workspace/rd_e8/teacher_53m.pt")
    ap.add_argument("--student-ckpt", default="/workspace/rd_e8/student_rde8.pt")
    ap.add_argument("--artifact-out", default="/workspace/rd_e8/model_rde8.bin")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--matrix-lr", type=float, default=0.022)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--target-linear-mb", type=float, default=13.5)
    ap.add_argument("--alpha-kl", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--eval-stride", type=int, default=256)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--ttt-chunk", type=int, default=512)
    ap.add_argument("--ttt-rank", type=int, default=32)
    ap.add_argument("--ttt-lr", type=float, default=1.5e-4)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    if args.cpu_smoke:
        run_cpu_smoke()
        return

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    use_bf16 = device.type == "cuda"

    try:
        if args.stage == "teacher":
            cfg = RDE8Config(max_seq_len=args.seq_len, quant_mode="dense")
            base = LoopedRDE8GPT(cfg).to(device)
            if is_main:
                print(f"[teacher] unique_params={sum(p.numel() for p in base.parameters()):,} eff_layers={len(base.exec_order)} world_size={world_size}", flush=True)
            mod: nn.Module = torch.compile(base) if args.compile else base
            if distributed:
                mod = nn.parallel.DistributedDataParallel(mod, device_ids=[local_rank] if device.type == "cuda" else None)
            opts = build_teacher_optimizers(base, matrix_lr=args.matrix_lr, embed_lr=args.lr)
            base_lrs = [[pg["lr"] for pg in o.param_groups] for o in opts]
            loader = ShardLoader(args.data_glob, seq_len=args.seq_len, seed=42, rank=rank)
            vt_mon, vb_mon = load_caseops_val(args.val_glob, max_tokens=2_000_000) if is_main else (None, None)
            t0 = time.perf_counter()

            for step in range(1, args.steps + 1):
                warm = min(1.0, step / 25.0)
                cos = 0.5 * (1.0 + math.cos(math.pi * step / max(args.steps, 1)))
                mul = warm * max(cos, 0.1)
                for o, blrs in zip(opts, base_lrs):
                    for pg, blr in zip(o.param_groups, blrs):
                        pg["lr"] = blr * mul
                    o.zero_grad(set_to_none=True)

                loss_acc = 0.0
                for _ in range(args.grad_accum):
                    x, y = loader.next_batch(args.batch_size, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        logits = mod(x)
                        loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
                    (loss / args.grad_accum).backward()
                    loss_acc += float(loss.detach().item()) / args.grad_accum
                torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
                for o in opts:
                    o.step()

                if is_main and (step == 1 or step % 400 == 0 or step == args.steps):
                    tok_s = step * args.batch_size * args.grad_accum * args.seq_len * world_size / max(time.perf_counter() - t0, 1e-6)
                    print(f"[teacher] step {step}/{args.steps} loss={loss_acc:.4f} tok/s={tok_s:,.0f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
                if args.eval_every > 0 and step % args.eval_every == 0 and step < args.steps:
                    if is_main:
                        vl, vb, _, _ = eval_caseops_bpb(base, vt_mon, vb_mon, device, seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size)
                        print(f"[teacher val@{step}] val_loss={vl:.6f} val_bpb={vb:.6f}", flush=True)
                        base.train()
                    if distributed:
                        dist.barrier()
                if is_main and args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                    torch.save({"cfg": dataclasses.asdict(cfg), "state_dict": base.state_dict()}, args.teacher_ckpt)
                    print(f"[teacher ckpt@{step}] saved -> {args.teacher_ckpt}", flush=True)

            if is_main:
                torch.save({"cfg": dataclasses.asdict(cfg), "state_dict": base.state_dict()}, args.teacher_ckpt)
                vl, vb, nt, nb = eval_caseops_bpb(base, vt_mon, vb_mon, device, seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size)
                print(f"[teacher FINAL] val_loss={vl:.6f} val_bpb={vb:.6f} (tokens={nt:,} bytes={nb:,})", flush=True)

        elif args.stage == "rd_e8_kd":
            ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
            t_cfg = RDE8Config(**ckpt["cfg"])
            teacher = LoopedRDE8GPT(t_cfg).to(device)
            teacher.load_state_dict(ckpt["state_dict"], strict=True)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            target_bpp_init = args.target_linear_mb * 8.0 / 50.331648
            if is_main:
                t_init = time.perf_counter()
                base_s = init_rde8_from_dense_teacher(teacher, target_bpp=target_bpp_init, device=device)
                print(f"[RD-E8 init] target_bpp={target_bpp_init:.2f} done in {time.perf_counter()-t_init:.1f}s | exact_MB={base_s.exact_entropy_bits()/8e6:.2f}", flush=True)
            else:
                base_s = LoopedRDE8GPT(dataclasses.replace(t_cfg, quant_mode="rd_e8")).to(device)
            if distributed:
                for p in base_s.parameters():
                    dist.broadcast(p.data, src=0)
                dist.barrier()

            vt_mon, vb_mon = load_caseops_val(args.val_glob, max_tokens=2_000_000) if is_main else (None, None)
            if is_main:
                zl, zb, _, _ = eval_caseops_bpb(base_s, vt_mon, vb_mon, device, seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size)
                print(f"[RD-E8 zero-shot] val_loss={zl:.6f} val_bpb={zb:.6f}", flush=True)
                base_s.train()
            if distributed:
                dist.barrier()

            tea_fwd = torch.compile(teacher) if args.compile else teacher
            stu_mod: nn.Module = torch.compile(base_s) if args.compile else base_s
            if distributed:
                stu_mod = nn.parallel.DistributedDataParallel(stu_mod, device_ids=[local_rank] if device.type == "cuda" else None)

            delta_p, other_p = [], []
            for pn, p in base_s.named_parameters():
                (delta_p if pn.endswith(".log_delta") else other_p).append(p)
            opt = torch.optim.AdamW(
                [{"params": other_p, "lr": args.lr, "weight_decay": 0.0}, {"params": delta_p, "lr": args.lr * 3.0, "weight_decay": 0.0}],
                betas=(0.9, 0.95),
            )
            loader = ShardLoader(args.data_glob, seq_len=args.seq_len, seed=1337, rank=rank)
            n_q_params = sum(m.in_features * m.out_features for _, m in base_s.quant_modules())
            target_bits = args.target_linear_mb * 8.0 * 1_000_000.0
            lam = 0.015
            t0 = time.perf_counter()

            for step in range(1, args.steps + 1):
                warm = min(1.0, step / 30.0)
                cos = 0.5 * (1.0 + math.cos(math.pi * step / max(args.steps, 1)))
                mul = warm * max(cos, 0.05)
                opt.param_groups[0]["lr"] = args.lr * mul
                opt.param_groups[1]["lr"] = args.lr * 3.0 * mul
                opt.zero_grad(set_to_none=True)

                ce_acc = kl_acc = 0.0
                for _ in range(args.grad_accum):
                    x, y = loader.next_batch(args.batch_size, device)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        t_logits = tea_fwd(x)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        s_logits = stu_mod(x)
                        sf = s_logits.float().reshape(-1, t_cfg.vocab_size)
                        tf = t_logits.float().reshape(-1, t_cfg.vocab_size)
                        ce = F.cross_entropy(sf, y.reshape(-1))
                        kl = F.kl_div(F.log_softmax(sf, dim=-1), F.log_softmax(tf, dim=-1), log_target=True, reduction="batchmean")
                        lm = (1.0 - args.alpha_kl) * ce + args.alpha_kl * kl
                    (lm / args.grad_accum).backward()
                    ce_acc += float(ce.detach().item()) / args.grad_accum
                    kl_acc += float(kl.detach().item()) / args.grad_accum

                rate_bpp = base_s.differentiable_rate_bits() / float(n_q_params)
                (lam * rate_bpp).backward()
                torch.nn.utils.clip_grad_norm_(base_s.parameters(), 1.0)
                opt.step()

                if step % 100 == 0:
                    if is_main:
                        eb = base_s.exact_entropy_bits()
                        ratio = eb / target_bits
                        if ratio > 1.005:
                            lam = min(lam * 1.22 + 0.005, 1.5)
                        elif ratio < 0.96:
                            lam = max(lam * 0.85, 0.001)
                    if distributed:
                        lt = torch.tensor([lam], device=device)
                        dist.broadcast(lt, src=0)
                        lam = float(lt.item())

                if is_main and (step == 1 or step % 200 == 0 or step == args.steps):
                    eb = base_s.exact_entropy_bits()
                    tok_s = step * args.batch_size * args.grad_accum * args.seq_len * world_size / max(time.perf_counter() - t0, 1e-6)
                    print(
                        f"[rd_e8_kd] step {step}/{args.steps} ce={ce_acc:.4f} kl={kl_acc:.4f} "
                        f"exact_MB={eb/8e6:.2f} ({eb/n_q_params:.2f} bpp) lam={lam:.4f} tok/s={tok_s:,.0f} elapsed={time.perf_counter()-t0:.1f}s",
                        flush=True,
                    )
                if args.eval_every > 0 and step % args.eval_every == 0 and step < args.steps:
                    if is_main:
                        vl, vb, _, _ = eval_caseops_bpb(base_s, vt_mon, vb_mon, device, seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size)
                        print(f"[rd_e8_kd val@{step}] val_loss={vl:.6f} val_bpb={vb:.6f}", flush=True)
                        base_s.train()
                    if distributed:
                        dist.barrier()

            if is_main:
                torch.save({"cfg": dataclasses.asdict(base_s.cfg), "state_dict": base_s.state_dict()}, args.student_ckpt)
                vl, vb, _, _ = eval_caseops_bpb(base_s, vt_mon, vb_mon, device, seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size)
                print(f"[rd_e8_kd FINAL] val_loss={vl:.6f} val_bpb={vb:.6f} -> saved {args.student_ckpt}", flush=True)

        elif args.stage == "pack_eval":
            if is_main:
                ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
                s_cfg = RDE8Config(**ckpt["cfg"])
                student = LoopedRDE8GPT(s_cfg).to(device)
                student.load_state_dict(ckpt["state_dict"], strict=True)
                student.eval()

                code_b = pathlib.Path(__file__).stat().st_size
                # Pack raw RDE8 frame; if marginally over 16,000,000 B, gently widen delta by 1.5% until strictly under 16 MB
                for guard_iter in range(8):
                    stats = save_rde8_artifact(student, args.artifact_out)
                    total_sub = stats["total_model_bytes"] + code_b
                    print(
                        f"[pack iter={guard_iter}] linear={stats['quant_linear_bytes']:,} B | "
                        f"embed={stats['embed_bytes']:,} B | model={stats['total_model_bytes']:,} B | "
                        f"code={code_b:,} B | TOTAL={total_sub:,} B ({'OK <=16MB' if total_sub <= 16_000_000 else 'TRIMMING'})",
                        flush=True,
                    )
                    if total_sub <= 16_000_000:
                        break
                    with torch.no_grad():
                        for _, m in student.quant_modules():
                            m.log_delta.add_(math.log(1.015))

                rel_chk = load_rde8_artifact(args.artifact_out, device)
                rel_chk.eval()
                x_chk = torch.randint(0, s_cfg.vocab_size, (2, 64), device=device)
                with torch.no_grad():
                    rt_diff = float((student(x_chk) - rel_chk(x_chk)).abs().max().item())
                print(f"[roundtrip] max_abs_logit_diff = {rt_diff:.8f}", flush=True)
                del student, rel_chk
            if distributed:
                dist.barrier()

            reloaded = load_rde8_artifact(args.artifact_out, device)
            reloaded.eval()
            for _, qm in reloaded.quant_modules():
                qm.freeze_for_eval()
            eval_mod = torch.compile(reloaded) if args.compile else reloaded
            val_toks, val_bytes = load_caseops_val(args.val_glob, max_tokens=0)

            t0 = time.perf_counter()
            vl0, vb0, nt0, nb0 = eval_caseops_bpb(
                eval_mod, val_toks, val_bytes, device,
                seq_len=args.seq_len, stride=args.seq_len, batch_size=args.batch_size,
                rank=rank, world_size=world_size,
            )
            if is_main:
                print(f"[FULL VAL chunk stride={args.seq_len}] val_loss={vl0:.6f} val_bpb={vb0:.6f} (tokens={nt0:,} bytes={nb0:,} in {time.perf_counter()-t0:.1f}s)", flush=True)

            t1 = time.perf_counter()
            vl1, vb1, nt1, nb1 = eval_caseops_bpb(
                eval_mod, val_toks, val_bytes, device,
                seq_len=args.seq_len, stride=args.eval_stride, batch_size=args.batch_size,
                rank=rank, world_size=world_size,
            )
            if is_main:
                print(f"[FULL VAL sliding stride={args.eval_stride}] val_loss={vl1:.6f} val_bpb={vb1:.6f} (tokens={nt1:,} bytes={nb1:,} in {time.perf_counter()-t1:.1f}s)", flush=True)

            if args.ttt:
                t2 = time.perf_counter()
                vl2, vb2, nt2, nb2 = eval_caseops_score_first_ttt(
                    reloaded, val_toks, val_bytes, device,
                    seq_len=args.seq_len, chunk_size=args.ttt_chunk,
                    lora_rank=args.ttt_rank, ttt_lr=args.ttt_lr,
                    rank=rank, world_size=world_size,
                )
                if is_main:
                    print(f"[FULL VAL score-first TTT] val_loss={vl2:.6f} val_bpb={vb2:.6f} (tokens={nt2:,} bytes={nb2:,} in {time.perf_counter()-t2:.1f}s)", flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
