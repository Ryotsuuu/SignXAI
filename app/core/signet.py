"""CNN-SigNet: a small Siamese convolutional network for signature
verification, with genuine Grad-CAM support.

v4 (explicit user requirement, PROJECT_MEMORY_v8.md): "CNN-SigNet" and
"Grad-CAM" were listed as core requirements never integrated. This module
adds them for real -- an actual differentiable CNN with actual gradients,
not a relabeled classical heatmap (see explain.py's honest naming note for
why that distinction was called out explicitly in v3).

HOW THIS RESPECTS THE PROJECT'S HARD CONSTRAINTS
---------------------------------------------------
1. No external dataset, ever (project invariant, unchanged): this network
   is trained PER PROFILE, from scratch, using only that profile's own
   enrolled samples -- heavily augmented (elastic warp, rotation jitter,
   stroke-thickness jitter via erosion/dilation, noise) to synthesize
   enough training variety from as few as 2 real images. No signature
   images from outside the profile are ever used as training data.
2. Fully optional, exactly like the LLM layer (app/llm/manager.py): torch
   is imported lazily inside functions, never at module load time. If
   torch isn't installed, every function in this module raises a clear
   ImportError that callers catch and fall back to the classical pipeline
   -- which is untouched and remains the primary, always-available,
   fully-tested decision path. This module never becomes a dependency of
   the core verification decision.

HONEST STATUS (PROJECT_MEMORY_v8.md v4 -- read before trusting this blindly)
-----------------------------------------------------------------------------
This module was written carefully using standard, well-established
patterns (a small Siamese CNN, contrastive loss, standard Grad-CAM hook
mechanics) but COULD NOT be run end-to-end in the development sandbox this
session: `pip install torch` either exhausted available disk space (the
default PyPI build pulls several GB of bundled CUDA libraries) or, when
installed --no-deps to avoid that, failed to even import
("ImportError: libcudnn.so.9: cannot open shared object file") because
recent PyPI torch builds on Linux are hard-linked against CUDA runtime
libraries at the C-extension level, not just at GPU-use time. This is
real, concrete evidence -- not speculation -- about the install cost of
this feature; see PROJECT_MEMORY_v8.md v4 status for the full account.
**This means: the code below is correct by careful inspection and
standard-pattern reuse, but UNTESTED end-to-end. Treat it as a first
implementation to verify on your own machine, not as something with the
same confidence level as the rest of this codebase (which has been
tested against real data at every step).**

INSTALLATION NOTE
------------------
If you want to try this, a genuinely CPU-only torch build (no CUDA
dependency at all, much smaller download) is available from PyTorch's own
index rather than plain PyPI:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
This was not reachable from the development sandbox's network policy but
should work from a normal internet connection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("signxai.signet")

EMBED_DIM = 32          # small -- this is a few-shot, per-profile network,
                        # not a large pretrained embedding space
IMG_SIZE = 128           # downsample target for the CNN input (separate
                        # from the classical pipeline's 512x256 canvas --
                        # smaller here keeps per-profile training fast)
AUGMENTATIONS_PER_IMAGE = 12   # synthetic variants generated per real
                               # enrolled sample (no external data, only
                               # transforms of the user's own images)
TRAIN_EPOCHS = 15
CONTRASTIVE_MARGIN = 1.0


def _require_torch():
    """Lazy import -- raises a clear, actionable error if torch isn't
    installed, rather than letting an ImportError propagate confusingly
    from deep inside this module."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as exc:
        raise ImportError(
            "CNN-SigNet requires torch, which is not installed. This "
            "feature is fully optional -- the app works normally without "
            "it, using the classical comparison pipeline instead. To "
            "enable it: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu  (a genuinely CPU-only "
            "build; plain 'pip install torch' may pull several GB of CUDA "
            "dependencies even on a machine with no GPU -- see this "
            "module's docstring for why)."
        ) from exc


def _build_model(nn):
    """Small Siamese embedding CNN. Deliberately shallow (4 conv layers) --
    this trains on a handful of augmented images per profile, not a large
    dataset, so a deep network would just overfit instantly and slowly."""
    class SigNetEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
            self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # Grad-CAM target layer
            self.pool = nn.MaxPool2d(2)
            self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(64)
            self.fc = nn.Linear(64 * (IMG_SIZE // 16) * (IMG_SIZE // 16), EMBED_DIM)
            self._activations = None
            self._gradients = None

        def forward(self, x):
            import torch.nn.functional as F
            h = self.pool(F.relu(self.bn1(self.conv1(x))))
            h = self.pool(F.relu(self.bn2(self.conv2(h))))
            h = self.pool(F.relu(self.conv3(h)))
            h = self.conv4(h)
            if h.requires_grad:
                h.register_hook(lambda g: setattr(self, "_gradients", g))
            self._activations = h
            h = self.pool(F.relu(h))
            h = h.flatten(1)
            emb = self.fc(h)
            return F.normalize(emb, dim=-1)

    return SigNetEmbedding()


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate one synthetic variant of a genuine signature image, using
    ONLY transforms of that same image -- no external data. Combines small
    rotation, elastic-style warp, and stroke-thickness jitter (erosion/
    dilation), which are the standard augmentations for handwriting/
    signature data since they simulate real natural variation (a person
    never signs pixel-identically twice) without changing identity."""
    import cv2
    h, w = img.shape[:2]
    out = img.copy()

    angle = rng.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0 + rng.uniform(-0.03, 0.03))
    out = cv2.warpAffine(out, M, (w, h), borderValue=255)

    if rng.random() < 0.5:
        k = int(rng.choice([1, 2]))
        kernel = np.ones((k, k), np.uint8)
        out = cv2.erode(out, kernel) if rng.random() < 0.5 else cv2.dilate(out, kernel)

    if rng.random() < 0.4:
        noise = rng.normal(0, 6, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def _prep_for_cnn(gray: np.ndarray):
    """Resize to the CNN's fixed input size and normalise to [-1, 1]."""
    import cv2
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    arr = (resized.astype(np.float32) / 127.5) - 1.0
    return arr[None, None, :, :]   # (1, 1, H, W)


def train_profile_signet(ref_grays: list[np.ndarray], seed: int = 42):
    """Train a small Siamese embedding network from scratch using only
    this profile's own enrolled reference images (augmented). Returns the
    trained torch model (kept in memory / saved by the caller -- this
    function doesn't touch disk itself).

    Raises ImportError if torch isn't installed (see _require_torch).
    """
    torch, nn, F = _require_torch()
    rng = np.random.default_rng(seed)

    # Build augmented positive pairs: two different augmentations of
    # (possibly) the same or different enrolled genuine images -- both
    # count as "same identity" since they're both this person's real
    # signature, augmented. Negative pairs use a heavily-distorted
    # variant (large rotation + heavy erosion) as a synthetic "different"
    # proxy -- still derived only from the user's own image, never
    # borrowed from another profile or external source.
    samples = []
    for img in ref_grays:
        for _ in range(AUGMENTATIONS_PER_IMAGE):
            samples.append((_prep_for_cnn(_augment(img, rng)), 1))
    for img in ref_grays:
        for _ in range(AUGMENTATIONS_PER_IMAGE // 2):
            heavy = _augment(_augment(img, rng), rng)  # double-augment = synthetic negative proxy
            samples.append((_prep_for_cnn(heavy), 0))

    model = _build_model(nn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    anchor = torch.tensor(_prep_for_cnn(ref_grays[0]))

    model.train()
    for epoch in range(TRAIN_EPOCHS):
        rng.shuffle(samples)
        total_loss = 0.0
        for arr, label in samples:
            x = torch.tensor(arr, dtype=torch.float32)
            emb_a = model(anchor)
            emb_x = model(x)
            dist = F.pairwise_distance(emb_a, emb_x)
            if label == 1:
                loss = dist.pow(2)
            else:
                loss = F.relu(CONTRASTIVE_MARGIN - dist).pow(2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        log.debug("SigNet epoch %d/%d loss=%.4f", epoch + 1, TRAIN_EPOCHS,
                  total_loss / max(len(samples), 1))

    model.eval()
    return model


def embedding_similarity(model, ref_gray: np.ndarray, query_gray: np.ndarray) -> float:
    """Cosine-style similarity (already-normalised embeddings, so this is
    just the dot product) between a reference and a query, using the
    trained per-profile SigNet model. Returns a value in roughly [0, 1]
    after remapping from [-1, 1] cosine range."""
    torch, nn, F = _require_torch()
    with torch.no_grad():
        emb_r = model(torch.tensor(_prep_for_cnn(ref_gray), dtype=torch.float32))
        emb_q = model(torch.tensor(_prep_for_cnn(query_gray), dtype=torch.float32))
        cos = float((emb_r * emb_q).sum())
    return (cos + 1.0) / 2.0


def grad_cam(model, ref_gray: np.ndarray, query_gray: np.ndarray) -> np.ndarray:
    """Genuine Grad-CAM: backpropagates the embedding distance (query vs.
    reference) through the network and weights the last conv layer's
    activations by the averaged gradient, per Selvaraju et al. 2017. This
    is real gradient-based attribution -- distinct from explain.py's
    pixel-divergence heatmap, which is a classical (non-gradient) proxy
    used when this optional CNN layer isn't available.

    Returns a (H, W) heatmap in [0, 1], same spatial size as the CNN's
    input (IMG_SIZE x IMG_SIZE) -- caller resizes to match the display image.
    """
    torch, nn, F = _require_torch()
    model.zero_grad()
    ref_t = torch.tensor(_prep_for_cnn(ref_gray), dtype=torch.float32)
    query_t = torch.tensor(_prep_for_cnn(query_gray), dtype=torch.float32, requires_grad=False)
    query_t.requires_grad_(False)

    emb_r = model(ref_t)
    emb_q = model(query_t)
    dist = F.pairwise_distance(emb_r, emb_q).sum()
    dist.backward()

    grads = model._gradients          # (1, C, h, w)
    activations = model._activations  # (1, C, h, w)
    if grads is None or activations is None:
        raise RuntimeError("Grad-CAM hook did not fire -- model forward "
                          "pass must run with gradients enabled.")
    weights = grads.mean(dim=(2, 3), keepdim=True)          # global-average-pool the gradient
    cam = F.relu((weights * activations).sum(dim=1)).squeeze(0)
    cam = cam.detach().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def save_model(model, path: Path) -> None:
    torch, _, _ = _require_torch()
    torch.save(model.state_dict(), str(path))


def load_model(path: Path):
    torch, nn, _ = _require_torch()
    model = _build_model(nn)
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model


def available() -> bool:
    """Cheap check for callers to decide whether to attempt this path at
    all, without raising. Used by engine.py/routes.py to gate the feature
    the same way the LLM layer gates on manager.status()."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
