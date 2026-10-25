import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamcdf.divergence import divergence_metrics
from streamcdf.model import Block, StreamCDF, build_causal_mask
from streamcdf.transport import d_kappa_per_clip, sinkhorn_divergence_loss


@pytest.fixture(autouse=True)
def limit_threads():
    # Tiny CPU grids are faster and more predictable without many worker threads.
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_reference_divergences_miss_proportional_count_error():
    truth = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 2, 2)
    scores = divergence_metrics(truth * 100, truth)
    assert all(np.abs(value).max() < 1e-6 for value in scores.values())
    assert truth.sum() * 100 != truth.sum()


def test_reference_js_is_not_bounded_by_standard_js_limit():
    truth = torch.tensor([1.0, 0.001]).reshape(1, 1, 1, 1, 2)
    pred = torch.tensor([0.001, 1000.0]).reshape(1, 1, 1, 1, 2)
    assert divergence_metrics(pred, truth)["adjs"][0] > np.log(2)


def test_future_observed_chunks_cannot_influence_past_tokens():
    torch.manual_seed(1)
    block = Block(dim=16, heads=2).eval()
    inputs = torch.randn(1, 6, 16)
    changed = inputs.clone()
    changed[:, 2:] += 100 * torch.randn_like(changed[:, 2:])
    mask = build_causal_mask(3, 2)
    original, _ = block(inputs, mask)
    perturbed, _ = block(changed, mask)
    assert torch.allclose(original[:, :2], perturbed[:, :2], atol=1e-6)


def test_zero_initialized_relu_model_starts_at_persistence():
    model = StreamCDF(img_size=8, patch=4, enc_tubelet=4, dec_tubelet=2,
                      obs_frames=8, pred_frames=12, dim=16, heads=2,
                      depth=1, dec_depth=1, out_act="relu").eval()
    observed = torch.rand(1, 1, 8, 8, 8)
    with torch.no_grad():
        result = model(observed)
    assert result.shape == (1, 1, 12, 8, 8)
    assert torch.equal(result, observed[:, :, -1:].expand_as(result))


def test_batched_transport_agrees_with_individual_solves_and_decomposes():
    torch.manual_seed(3)
    pred = torch.rand(3, 1, 2, 4, 4)
    truth = torch.rand_like(pred)
    config = dict(den_factor=1, pool=1, eps=0.5, iters=40, kappa=2)
    batched = d_kappa_per_clip(pred, truth, batch_chunk=3, decompose=True, **config)
    individual = d_kappa_per_clip(pred, truth, batch_chunk=1, decompose=True, **config)
    for left, right in zip(batched, individual):
        assert np.allclose(left, right, rtol=1e-9, atol=1e-9)
    assert np.allclose(batched[0], batched[1] + 2 * batched[2])


def test_debiased_training_loss_is_zero_for_identical_maps():
    density = torch.rand(1, 1, 1, 4, 4, requires_grad=True)
    loss, _, _ = sinkhorn_divergence_loss(density, density.detach(),
                                         den_factor=1, pool=1, eps=0.5, iters=40, kappa=2)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert torch.isfinite(density.grad).all()
