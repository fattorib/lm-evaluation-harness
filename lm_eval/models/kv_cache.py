"""Key-Value cache for easy operability."""
from typing import Tuple

import torch


class KVCache:
    max_kv_size: int
    num_heads: int
    head_dim: int
    keys: torch.Tensor
    values: torch.Tensor
    device: str
    current_kv_size: int = 0

    def __init__(self, max_kv_size, num_heads, head_dim, device):
        self.keys = torch.full(
            (1, num_heads, max_kv_size, head_dim),
            fill_value=0.0,
            dtype=torch.float16,
            device=device,
        )
        self.values = torch.full(
            (1, num_heads, max_kv_size, head_dim),
            fill_value=0.0,
            dtype=torch.float16,
            device=device,
        )

        self.max_kv_size = max_kv_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

    def update_kv(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """Updates the KV cache with new keys and values."""
        nb, nh, kv, hd = (
            new_keys.shape[0],
            new_keys.shape[1],
            new_keys.shape[2],
            new_keys.shape[3],
        )

        assert nh == self.num_heads
        assert hd == self.head_dim

        indices = torch.arange(
            self.current_kv_size, self.current_kv_size + kv, device=self.device
        )
        self.keys[:, :, indices, :] = new_keys.to(self.keys.dtype)
        self.values[:, :, indices, :] = new_values.to(self.keys.dtype)
        self.current_kv_size += kv

    def get_valid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of the valid (i.e. unpadded) keys and values as tensor views."""

        return (
            self.keys[:, :, : self.current_kv_size, :],
            self.values[:, :, : self.current_kv_size, :],
        )

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns a tuple of all keys and values and the count of the valid keys and values."""
        return self.keys, self.values, self.current_kv_size

    def __repr__(self) -> str:
        return f"KVCache: current_kv_size={self.current_kv_size}, max_kv_size={self.max_kv_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"
