import numpy as np
import torch

from transformercvn.network.layers.packed_data import (
    masked_pack_1d_precomputed,
    masked_pad_1d_precomputed,
)
from transformercvn.network.networks.neutrino_full_dense_network import NeutrinoDenseNetwork
from transformercvn.network.trainers.neutrino_full_base_trainer import tpr_at_fpr_ovr_last_class
from transformercvn.options import Options


def test_masked_pack_roundtrip():
    values = torch.arange(24.0).reshape(2, 3, 4)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    packed, rows, cols = masked_pack_1d_precomputed(values, mask)
    restored = masked_pad_1d_precomputed(packed, rows, cols, 2, 3)
    torch.testing.assert_close(restored[mask], values[mask])
    assert torch.all(restored[~mask] == 0)


def test_dense_forward_backward_without_sparse_dependencies():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    options = Options()
    options.update_options(
        {
            "hidden_dim": 16,
            "initial_feature_dim": 8,
            "initial_pixel_dim": 8,
            "feature_embedding_dim": 8,
            "pixel_embedding_dim": 8,
            "position_embedding_dim": 8,
            "num_encoder_layers": 1,
            "num_attention_heads": 2,
            "num_prong_decoder_layers": 1,
            "densenet_structure": [1],
            "densenet_growth_rate": 4,
        }
    )
    model = NeutrinoDenseNetwork(options, 3, 2, 1, 2, 4)
    event_logits, prong_logits = model(
        torch.randn(2, 2, 3),
        torch.randn(2, 2),
        torch.randn(2, 1, 16, 16),
        torch.ones(2, 1, dtype=torch.bool),
        torch.randn(3, 1, 16, 16),
        torch.tensor([[True, True], [True, False]]),
    )
    assert event_logits.shape == (2, 4)
    assert prong_logits.shape == (2, 2, 2)
    loss = event_logits.square().mean() + prong_logits.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.event_decoder is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_roc_preserves_historical_first_point_at_or_above_target():
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    assert tpr_at_fpr_ovr_last_class(targets, probabilities, 0.1) == 0.0
    assert tpr_at_fpr_ovr_last_class(np.zeros(4), probabilities, 0.1) == 0.0
