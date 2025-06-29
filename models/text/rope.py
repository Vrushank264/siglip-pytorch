import torch


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to the query and key states
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (query_states*cos) + (rotate_half(query_states)*sin)
    k_embed = (key_states*cos) + (rotate_half(key_states)*sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)