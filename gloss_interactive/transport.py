# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Websocket payload helpers for the Gloss server.

Binary (de)serialization is Kaolin's ``kaolin.visualize.web.io``. Its
``to_binary`` PNG-compresses image-shaped uint8 arrays by default (wire types
12/13), which the add-on's decoder does not understand, so :func:`to_binary`
pins ``image_format="raw"`` to keep the wire format the add-on expects.
"""

import numpy as np
import torch
from kaolin.visualize.web import io as kaolin_io

from gloss_interactive.protocol import DEFAULT_CHUNK_BYTES, DTYPE_FLOAT32, DTYPE_UINT8

from_binary = kaolin_io.from_binary


def to_binary(value):
    """Encode ``value`` with Kaolin's binary format, without image compression."""
    return kaolin_io.to_binary(value, image_format=kaolin_io.ImageFormat.RAW)


def split_tensor(tensor, max_chunk_bytes=1_000_000):
    """
    Returns a list of smaller tensors, each <= max_chunk_bytes.
    """
    bytes_per_elem = tensor.element_size()        # e.g., float32 = 4 bytes
    total_elems = tensor.numel()
    elems_per_chunk = max_chunk_bytes // bytes_per_elem
    elems_per_chunk = max(1, elems_per_chunk)

    flat = tensor.contiguous().view(-1)

    chunks = [
        flat[i:i+elems_per_chunk].clone()
        for i in range(0, total_elems, elems_per_chunk)
    ]

    return chunks


def encode_pixels(image, dtype=DTYPE_UINT8):
    """Quantize a ``[0, 1]`` pixel tensor for transport.

    Quantization happens on-device before the host copy, so a 4K RGBA texture
    crosses the PCIe bus as 64 MB rather than 256 MB, and again as 64 MB on the
    wire. uint8 is lossless for the 8-bit basecolor data both ends store.

    Args:
        image: Torch tensor or NumPy array of pixels, nominally in ``[0, 1]``.
        dtype: One of ``DTYPE_UINT8`` / ``DTYPE_FLOAT32``.

    Returns:
        tuple[np.ndarray, str]: The array to put on the wire and its dtype tag.
    """
    if isinstance(image, torch.Tensor):
        if dtype == DTYPE_UINT8:
            quantized = (image.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            return quantized.cpu().numpy(), DTYPE_UINT8
        return image.detach().to(torch.float32).cpu().numpy(), DTYPE_FLOAT32

    image = np.asarray(image)
    if dtype == DTYPE_UINT8:
        return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), DTYPE_UINT8
    return image.astype(np.float32), DTYPE_FLOAT32


def decode_pixels(image, dtype=DTYPE_FLOAT32):
    """Inverse of :func:`encode_pixels`; returns a float32 tensor in ``[0, 1]``."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if dtype == DTYPE_UINT8 or image.dtype == torch.uint8:
        return image.to(torch.float32) / 255.0
    return image.to(torch.float32)


def send_large_image(name: str, task_name: str, image, chunk_size: int = DEFAULT_CHUNK_BYTES,
                     msg_type: str = "image", dtype: str = DTYPE_UINT8, extra: dict = None):

    """
    Send a large image through WebSocket in multiple binary chunks.

    Parameters:
        name        : image name string
        task_name   : server task identifier stored in each message
        image       : pixel tensor/array in [0, 1]
        chunk_size  : size in bytes (default: 10MB)
        msg_type    : routing tag written into every chunk's ``type`` field
        dtype       : wire dtype; see :func:`encode_pixels`
        extra       : additional key/values copied into every chunk message
    """
    payload, dtype_tag = encode_pixels(image, dtype=dtype)
    chunks = split_tensor(torch.from_numpy(payload), max_chunk_bytes=chunk_size)
    total_chunks = len(chunks)
    msgs = []
    for idx in range(total_chunks):
        chunk = chunks[idx]
        msg = {
            "type": msg_type,
            "name": name,
            "task_name": task_name,
            "chunk_index": idx,
            "chunk_total": total_chunks,
            "dtype": dtype_tag,
            "image": chunk,
        }
        if extra:
            msg.update(extra)
        msgs.append(msg)
    return msgs
