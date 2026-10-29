#!/usr/bin/env python
"""
Step 5 (+ first version of step 6): ResNet-18 patch encoder and similarity-based localisation.

Usage:
    python model.py info                # model summary, parameter count, GPU and speed check
    python model.py test <index> [n]    # localise lesions of patient <index> with the UNTRAINED
                                        # (ImageNet-only) encoder and save heatmap images
"""
import importlib, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

ep = None
for _name in ("extract_patches_v3", "extract_patches"):
    try:
        ep = importlib.import_module(_name); break
    except ImportError:
        pass
if ep is None:
    sys.exit("extract_patches_v3.py not found next to model.py")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEncoder(nn.Module):
    """ResNet-18: 3 x 64 x 64 patch -> unit-length feature vector."""
    def __init__(self, pretrained=True, embed_dim=128, drop_maxpool=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if drop_maxpool:
            net.maxpool = nn.Identity()          # keep a 4x4 final map on 64x64 input
        net.fc = nn.Identity()                   # 512-d features instead of class scores
        self.backbone = net
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, embed_dim))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x, use_head=True):
        x = (x - self.mean) / self.std
        f = self.backbone(x)
        if use_head:
            f = self.head(f)
        return F.normalize(f, dim=1)             # dot product = cosine similarity


@torch.no_grad()
def embed(model, patches, use_head=True, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(patches), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(patches[i:i + batch_size])).to(DEVICE)
        out.append(model(x, use_head=use_head).cpu())
    return torch.cat(out)


def fine_offsets(half=4):
    o = np.arange(-half, half + 1)
    g = np.meshgrid(o, o, o, indexing="ij")
    return np.stack([a.ravel() for a in g], axis=1)


def localise(model, bl_img, fu_img, p_bl_mm, init_mm, use_head=True, refine=True):
    """Returns (predicted centre in mm, coarse heatmap indexed [x, y, z])."""
    query = embed(model, ep.baseline_patch(bl_img, p_bl_mm)[None], use_head)
    offs = ep.candidate_offsets()
    cands = ep.candidate_patches(ep.search_region(fu_img, init_mm), offs)
    sims = (embed(model, cands, use_head) @ query.T).squeeze(1).numpy()
    best = np.asarray(init_mm) + offs[sims.argmax()]
    n = len(np.unique(offs[:, 0]))
    heat = sims.reshape(n, n, n)
    if refine:
        f_offs = fine_offsets()
        f_cands = ep.candidate_patches(ep.search_region(fu_img, best), f_offs)
        f_sims = (embed(model, f_cands, use_head) @ query.T).squeeze(1).numpy()
        best = best + f_offs[f_sims.argmax()]
    return best, heat


def info():
    print(f"PyTorch {torch.__version__}, device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    model = PatchEncoder().to(DEVICE)
    n_back = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"Parameters: backbone {n_back / 1e6:.1f} M, head {n_head / 1e3:.0f} K")
    x = np.random.rand(len(ep.candidate_offsets()), 3, 64, 64).astype(np.float32)
    t0 = time.time()
    v = embed(model, x)
    print(f"Embedded {len(x)} patches -> {tuple(v.shape)} in {time.time() - t0:.1f} s")


def save_heatmap(heat, true_off, pred_off, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = heat.shape[0]
    to_idx = lambda o: (np.asarray(o) + ep.CUBE_HALF) / ep.GRID_STEP
    views = [("axial (x-y)", heat.max(axis=2).T, 0, 1, False),
             ("coronal (x-z)", heat.max(axis=1).T[::-1], 0, 2, True),
             ("sagittal (y-z)", heat.max(axis=0).T[::-1], 1, 2, True)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (name, img, a, b, flip) in zip(axes, views):
        im = ax.imshow(img, cmap="viridis")
        for off, color, label in [(true_off, "red", "true"), (pred_off, "white", "predicted")]:
            col, row = to_idx(off[a]), to_idx(off[b])
            if flip:
                row = n - 1 - row
            ax.scatter(col, row, c=color, marker="x", s=60, label=label)
        ax.scatter((n - 1) / 2, (n - 1) / 2, c="cyan", marker="+", s=60, label="initial estimate")
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7, loc="lower left")
    fig.colorbar(im, ax=axes, shrink=0.8, label="max cosine similarity")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def test(index, n_lesions=3):
    import SimpleITK as sitk
    pid = ep.patient_ids()[index]
    os.makedirs(ep.PATCH_DIR, exist_ok=True)
    print(f"Device: {DEVICE}. Patient {pid}. Untrained encoder (ImageNet features, no head).")
    model = PatchEncoder().to(DEVICE)
    df, affine = ep.lesion_list(pid)
    images, rows = {}, []

    def img(tp, i):
        if (tp, int(i)) not in images:
            images[(tp, int(i))] = sitk.ReadImage(ep.ct_path(pid, tp, i), sitk.sitkFloat32)
        return images[(tp, int(i))]

    for _, r in df.head(n_lesions).iterrows():
        bl, fu = img("BL", r["img_id_bl"]), img("FU", r["img_id_fu"])
        p_bl = ep.vox_to_mm(bl, ep.parse_xyz(r["cog_bl"]))
        p_true = ep.vox_to_mm(fu, ep.parse_xyz(r["cog_fu"]))
        init, source = ep.initial_estimate_mm(r, fu, affine)
        if init is None:
            continue
        t0 = time.time()
        pred, heat = localise(model, bl, fu, p_bl, init, use_head=False)
        e_init, e_pred = np.linalg.norm(init - p_true), np.linalg.norm(pred - p_true)
        rows.append((e_init, e_pred))
        print(f"  lesion {r['lesion_id']}: initial estimate {e_init:.1f} mm -> model {e_pred:.1f} mm "
              f"({time.time() - t0:.1f} s)")
        save_heatmap(heat, p_true - init, pred - init,
                     f"{pid} lesion {r['lesion_id']}: initial {e_init:.1f} mm, model {e_pred:.1f} mm (untrained)",
                     os.path.join(ep.PATCH_DIR, f"heatmap_{pid}_lesion{r['lesion_id']}.png"))
    if rows:
        e = np.array(rows)
        print(f"Median error: initial {np.median(e[:, 0]):.1f} mm, untrained model {np.median(e[:, 1]):.1f} mm")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "info":
        info()
    elif cmd == "test":
        test(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        print(__doc__)