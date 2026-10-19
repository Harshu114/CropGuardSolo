import time, torch
from model.architecture.v7_model import CropGuardMultiTaskV7
from model.training.losses import multitask_loss
from model.training.dataset import CropGuardDataset, collate
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropGuardMultiTaskV7(nc=8, pretrained=True).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

ds = CropGuardDataset("/home/logiclocus/tmp/cgpack/cropguard_v7_split", "train", 320, augment=True)
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)

print(f"Dataset length: {len(ds)} images. Total batches at batch_size=8: {len(loader)}")

it = iter(loader)
for _ in range(2):
    b = next(it)
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
    with torch.amp.autocast("cuda", enabled=True):
        out = model(b["image"], with_analysis=True, active_tasks={"det", "cls", "seg"})
        loss, _ = multitask_loss(out, b, 320, nc=8, reg_max=8, train_sev=False, active_tasks={"det", "cls", "seg"}, two_channel_seg=True)
    loss.backward()
    opt.step()
    opt.zero_grad()

torch.cuda.synchronize()
t0 = time.time()
n_batches = 15
for _ in range(n_batches):
    b = next(it)
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
    with torch.amp.autocast("cuda", enabled=True):
        out = model(b["image"], with_analysis=True, active_tasks={"det", "cls", "seg"})
        loss, _ = multitask_loss(out, b, 320, nc=8, reg_max=8, train_sev=False, active_tasks={"det", "cls", "seg"}, two_channel_seg=True)
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()
dt = time.time() - t0
step_time = dt / n_batches
epoch_time = step_time * len(loader)
print(f"Batch_size=8 step time (forward+backward+amp): {step_time*1000:.1f} ms")
print(f"Measured time per epoch: {epoch_time:.2f} seconds ({epoch_time/60:.2f} min)")
print(f"Projected total time for 30 epochs: {epoch_time*30/60:.1f} minutes")
