"""Properties of the architecture modules that hold by construction."""
import torch

from strata.manifold.decay import Recovery


def test_diffusion_consistent_decoder():
    """The decoder head keeps the decay form of the diffusion process gate by gate, whatever the latent sequence."""
    torch.manual_seed(0)
    decoder = Recovery(hidden_dim=16, output_dim=1, num_layers=1, monotone_head=True)
    with torch.no_grad():
        y = decoder(3.0 * torch.randn(8, 31, 16))
    assert y.shape == (8, 31, 1)
    assert torch.all(torch.isfinite(y))
    assert torch.all(torch.diff(y, dim=1) <= 1e-7)
