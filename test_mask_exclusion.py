import torch
import torch.nn.functional as F
from model.training.losses import seg_loss_v7, sev_loss_fn

def test_exclusion_unit():
    print("=== Testing seg_loss_v7 leaf exclusion ===")
    B, H, W = 2, 320, 320
    # Batch of 2: Sample 0 is normal, Sample 1 is excluded (leaf = -1.0)
    seg_logits = torch.randn(B, 2, H, W, requires_grad=True)
    lesion_mask = torch.zeros(B, 1, H, W)
    lesion_mask[0, :, 10:20, 10:20] = 1.0
    lesion_mask[1, :, 10:20, 10:20] = 1.0

    leaf_mask = torch.zeros(B, 1, H, W)
    leaf_mask[0, :, 5:50, 5:50] = 1.0
    leaf_mask[1, :, :, :] = -1.0  # Excluded sentinel

    loss = seg_loss_v7(seg_logits, lesion_mask, leaf_mask)
    print(f"Initial seg loss: {loss.item():.6f}")

    # Perturb the prediction for the excluded sample's leaf channel only
    seg_logits_perturbed = seg_logits.clone()
    with torch.no_grad():
        seg_logits_perturbed[1, 1, :, :] += 100.0

    loss_perturbed = seg_loss_v7(seg_logits_perturbed, lesion_mask, leaf_mask)
    print(f"Perturbed seg loss: {loss_perturbed.item():.6f}")
    diff = abs((loss - loss_perturbed).item())
    print(f"Leaf seg loss difference upon perturbing excluded leaf logits: {diff:.8f}")
    assert diff < 1e-6, f"Expected 0 difference, got {diff}"

    # Also test gradients on excluded leaf prediction
    loss.backward()
    grad_leaf_excluded = seg_logits.grad[1, 1].abs().sum().item()
    print(f"Gradient sum for excluded sample leaf prediction: {grad_leaf_excluded:.8f}")
    assert grad_leaf_excluded == 0.0, f"Expected gradient 0, got {grad_leaf_excluded}"
    assert not torch.isnan(seg_logits.grad).any(), "NaN found in gradients!"

    print("\n=== Testing sev_loss_fn exclusion ===")
    sev_cont = torch.randn(2, 1, requires_grad=True)
    sev_ord = torch.randn(2, 4, requires_grad=True)

    sev_cont_t = torch.tensor([0.05, -1.0])
    sev_ord_t = torch.tensor([1, -1])

    loss_sev = sev_loss_fn(sev_cont, sev_ord, sev_cont_t, sev_ord_t)
    print(f"Initial sev loss: {loss_sev.item():.6f}")

    # Perturb sample 1 predictions
    sev_cont_p = sev_cont.clone()
    sev_ord_p = sev_ord.clone()
    with torch.no_grad():
        sev_cont_p[1] += 50.0
        sev_ord_p[1] += 50.0

    loss_sev_p = sev_loss_fn(sev_cont_p, sev_ord_p, sev_cont_t, sev_ord_t)
    diff_sev = abs((loss_sev - loss_sev_p).item())
    print(f"Severity loss difference upon perturbing excluded sample: {diff_sev:.8f}")
    assert diff_sev < 1e-6, f"Expected 0 difference, got {diff_sev}"

    loss_sev.backward()
    assert (sev_cont.grad[1] == 0.0).all(), "Excluded sample got non-zero sev_cont grad!"
    assert (sev_ord.grad[1] == 0.0).all(), "Excluded sample got non-zero sev_ord grad!"
    assert not torch.isnan(sev_cont.grad).any(), "NaN found in sev_cont grad!"
    assert not torch.isnan(sev_ord.grad).any(), "NaN found in sev_ord grad!"

    print("\nALL EXCLUSION UNIT TESTS PASSED: Exactly 0 contribution and no NaNs.")

if __name__ == "__main__":
    test_exclusion_unit()
