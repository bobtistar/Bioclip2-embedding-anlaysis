"""Embedding extraction for BioCLIP2, BioCLIP, and OpenCLIP backbones.

Usage
-----
from extract_embeddings import load_model, encode_images, encode_texts

model, preprocess, tokenizer, info = load_model("openclip-vitb32")
img_emb = encode_images(model, preprocess, image_paths, device="cpu")
txt_emb = encode_texts(model, tokenizer, prompts, device="cpu")

Supported model keys
--------------------
  - "openclip-vitb32"  : open_clip ViT-B/32 (laion2b_s34b_b79k)        -- DEFAULT for toy runs
  - "openclip-vitl14"  : open_clip ViT-L/14 (laion2b_s32b_b82k)
  - "bioclip"          : imageomics/bioclip (HuggingFace), ViT-B/16
  - "bioclip2"         : imageomics/bioclip-2 (HuggingFace), ViT-L/14

For BioCLIP / BioCLIP2 the script requires HuggingFace hub access. If unavailable,
the loader raises a clear error and the caller should fall back to mock embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Optional, Tuple, Any
import os
import warnings

import numpy as np
import torch


@dataclass
class ModelInfo:
    name: str
    embedding_dim: int
    image_size: int
    backbone: str
    source: str  # 'open_clip' or 'hf'


def _torch_device(device: Optional[str]) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@contextmanager
def _hf_token_env(hf_token: Optional[str]):
    """Temporarily expose an HF token through the env vars used by huggingface_hub."""
    if not hf_token:
        yield
        return

    old_hf_token = os.environ.get("HF_TOKEN")
    old_hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        yield
    finally:
        if old_hf_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = old_hf_token
        if old_hub_token is None:
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        else:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hub_token


# ---------------------------------------------------------------------------
# OpenCLIP loaders
# ---------------------------------------------------------------------------

def _load_openclip(
    name: str,
    pretrained: str,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, Any, ModelInfo]:
    import open_clip

    with _hf_token_env(hf_token):
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(name)
    model.eval()
    # OpenCLIP encodes embedding dim implicitly; we probe with a dummy text
    with torch.no_grad():
        tok = tokenizer(["hello"]).long()
        emb = model.encode_text(tok)
    info = ModelInfo(
        name=f"openclip:{name}@{pretrained}",
        embedding_dim=int(emb.shape[-1]),
        image_size=224,
        backbone=name,
        source="open_clip",
    )
    return model, preprocess, tokenizer, info


# ---------------------------------------------------------------------------
# HuggingFace loader for BioCLIP variants
# ---------------------------------------------------------------------------

def _load_hf_bioclip(
    repo: str,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, Any, ModelInfo]:
    """Load BioCLIP or BioCLIP2 from HuggingFace. Requires network + open_clip 3.0+
    or transformers, depending on packaging.

    NOTE: As of writing, `imageomics/bioclip-2` is distributed as an open_clip-compatible
    checkpoint. We use open_clip's `create_model_from_pretrained` with HF URL.
    """
    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError(f"open_clip required: {e}")
    try:
        # open_clip supports hf-hub:org/repo syntax
        with _hf_token_env(hf_token):
            model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{repo}")
            tokenizer = open_clip.get_tokenizer(f"hf-hub:{repo}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {repo} from HuggingFace via open_clip. "
            f"Underlying error: {e}. "
            f"Check internet access, HF cache, and that the repo exists."
        )
    model.eval()
    with torch.no_grad():
        tok = tokenizer(["hello"]).long()
        emb = model.encode_text(tok)
    info = ModelInfo(
        name=f"hf:{repo}",
        embedding_dim=int(emb.shape[-1]),
        image_size=224,
        backbone=repo,
        source="hf",
    )
    return model, preprocess, tokenizer, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "openclip-vitb32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "openclip-vitl14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "bioclip": "imageomics/bioclip",
    "bioclip2": "imageomics/bioclip-2",
}


def load_model(
    key: str,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, Any, ModelInfo]:
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key {key}. Available: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[key]
    if isinstance(entry, tuple):
        name, pretrained = entry
        return _load_openclip(name, pretrained, hf_token=hf_token)
    return _load_hf_bioclip(entry, hf_token=hf_token)


class _PathImageDataset(torch.utils.data.Dataset):
    """파일 경로 리스트를 PIL.Image -> preprocess(tensor)로 변환하는 Dataset.

    DataLoader의 num_workers와 함께 쓰면 디스크 I/O와 GPU 계산을 겹쳐서 빨라진다.
    """

    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # truncated 다운로드 JPEG도 디코딩
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


@torch.no_grad()
def encode_images(
    model,
    preprocess,
    images,  # list of PIL.Image OR list of paths OR torch tensor (N, 3, H, W)
    device: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_amp: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """이미지 리스트를 인코딩하여 (N, D) numpy 배열로 반환.

    Args:
        images: PIL.Image 리스트, 파일 경로 리스트, 또는 미리 preprocess된 (N,3,H,W) tensor.
        device: 'cuda'/'mps'/'cpu'/None(자동 감지).
        batch_size: GPU에서 ViT-B/32는 64-128, ViT-L/14는 16-32 권장.
        num_workers: 경로 입력 시 DataLoader worker 수. Windows에서는 0으로 두는 게 안전.
        use_amp: CUDA에서 float16 mixed precision. 보통 1.5~2배 빠름.
        show_progress: tqdm 진행률 표시.
    """
    device_t = _torch_device(device)
    model = model.to(device_t)
    amp_enabled = bool(use_amp and device_t.type == "cuda")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore[misc]
            return it

    def _autocast_ctx():
        if amp_enabled:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        from contextlib import nullcontext
        return nullcontext()

    # Case 1: 이미 preprocess 된 tensor
    if isinstance(images, torch.Tensor):
        outs: List[np.ndarray] = []
        n_batches = (len(images) + batch_size - 1) // batch_size
        iterator = images.split(batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="encode_image", unit="batch")
        for b in iterator:
            b = b.to(device_t, non_blocking=True)
            with _autocast_ctx():
                feat = model.encode_image(b)
            outs.append(feat.float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    # Case 2: 파일 경로 리스트 → DataLoader 병렬 로드
    if len(images) > 0 and isinstance(images[0], str):
        dataset = _PathImageDataset(images, preprocess)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device_t.type == "cuda"),
        )
        outs = []
        iterator = tqdm(loader, total=len(loader), desc="encode_image", unit="batch") if show_progress else loader
        for tens in iterator:
            tens = tens.to(device_t, non_blocking=True)
            with _autocast_ctx():
                feat = model.encode_image(tens)
            outs.append(feat.float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    # Case 3: PIL.Image 리스트 (소수)
    from PIL import Image
    outs = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    indices = list(range(0, len(images), batch_size))
    if show_progress:
        indices = tqdm(indices, total=n_batches, desc="encode_image", unit="batch")
    for i in indices:
        batch = images[i:i + batch_size]
        tensors = []
        for x in batch:
            if isinstance(x, str):
                img = Image.open(x).convert("RGB")
            else:
                img = x
            tensors.append(preprocess(img))
        tens = torch.stack(tensors).to(device_t, non_blocking=True)
        with _autocast_ctx():
            feat = model.encode_image(tens)
        outs.append(feat.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    prompts: List[Optional[str]],
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode prompts. None values are encoded as zero vectors (matching dim).

    Returns np.ndarray of shape (N, D).
    """
    device_t = _torch_device(device)
    model = model.to(device_t)
    # Detect None entries
    is_none = [p is None for p in prompts]
    non_none = [p for p in prompts if p is not None]

    # Probe embedding dim by encoding a single token first
    if non_none:
        tok0 = tokenizer(["probe"]).long().to(device_t)
        D = int(model.encode_text(tok0).shape[-1])
    else:
        # all None -> return zeros (caller should expect)
        return np.zeros((len(prompts), 1), dtype=np.float32)

    out_full = np.zeros((len(prompts), D), dtype=np.float32)
    if non_none:
        feats: List[np.ndarray] = []
        for i in range(0, len(non_none), batch_size):
            batch = non_none[i:i + batch_size]
            tok = tokenizer(batch).long().to(device_t)
            feat = model.encode_text(tok)
            feats.append(feat.float().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        # scatter back to original positions
        j = 0
        for i, none_i in enumerate(is_none):
            if not none_i:
                out_full[i] = feats[j]
                j += 1
    return out_full


# ---------------------------------------------------------------------------
# Mock fallback (when models cannot load, e.g. no internet)
# ---------------------------------------------------------------------------

def mock_image_embeddings(
    labels: np.ndarray,
    n_class: int,
    dim: int = 256,
    intra_noise: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic embeddings where each class has its own centroid + noise.
    Useful for sanity-checking the analysis pipeline without a real model.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_class, dim))
    Z = centroids[labels] + intra_noise * rng.normal(0, 1, size=(len(labels), dim))
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z.astype(np.float32)


def mock_text_embeddings(
    prompts: List[Optional[str]],
    dim: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Hash prompts to a deterministic embedding for mock runs."""
    rng_seed = seed
    out = np.zeros((len(prompts), dim), dtype=np.float32)
    for i, p in enumerate(prompts):
        if p is None:
            continue
        h = abs(hash(p)) % (2 ** 32)
        rng_local = np.random.default_rng(h + rng_seed)
        v = rng_local.normal(0, 1, size=dim)
        v /= (np.linalg.norm(v) + 1e-12)
        out[i] = v
    return out


if __name__ == "__main__":
    # Smoke test: load OpenCLIP ViT-B/32 if available, otherwise mock.
    try:
        model, preprocess, tokenizer, info = load_model("openclip-vitb32")
        print("Loaded:", info)
        emb = encode_texts(model, tokenizer, ["a photo of a sparrow", "a photo of a rose"], device="cpu")
        print("Text emb shape:", emb.shape)
    except Exception as e:
        print(f"Real load failed ({e}); falling back to mock.")
        y = np.array([0, 0, 1, 1])
        z = mock_image_embeddings(y, n_class=2)
        print("Mock emb shape:", z.shape)
