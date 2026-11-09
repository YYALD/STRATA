"""Properties of the architecture modules that hold by construction."""
import torch

from strata.models.decay_manifold import Recovery


def test_monotone_decoder():
    """With its monotone head the decoder returns decays that do not rise in time, whatever the latent sequence."""
    torch.manual_seed(0)
    decoder = Recovery(hidden_dim=16, output_dim=1, num_layers=1, monotone_head=True)
    with torch.no_grad():
        y = decoder(3.0 * torch.randn(8, 31, 16))
    assert y.shape == (8, 31, 1)
    assert torch.all(torch.isfinite(y))
    assert torch.all(torch.diff(y, dim=1) <= 1e-7)
