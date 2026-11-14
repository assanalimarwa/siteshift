"""
Frozen foundation-model encoders for the encoder grid (model.py: arch "fm_<name>").

Every encoder takes a batch normalised with its own mean / std, at a side length that is a
multiple of its patch size, and returns the outputs of four evenly spaced transformer blocks as
B x C x h x w maps (class and register tokens dropped). Weights are never trained.

    uni         UNI (MahmoodLab/UNI), ViT-L/16, pathology, 100M patches       timm, gated (auto)
    virchow2    Virchow2 (paige-ai/Virchow2), ViT-H/14, pathology, 3.1M WSIs   timm, gated (auto)
    conch       CONCH vision trunk (MahmoodLab/CONCH), ViT-B/16 @448, pathology image-text    timm, gated (auto)
    dinov2      DINOv2 ViT-L/14 (facebook/dinov2-large), natural images                         transformers, open
    dinov3      DINOv3 ViT-L/16 distilled (facebook/dinov3-vitl16-pretrain-lvd1689m), natural images
                                                                            transformers, gated (manual)
    unimedclip  UniMed-CLIP ViT-L/14-336 (base text encoder), medical image-text CLIP; loaded with
                the checkpoint and repo of $SITESHIFT_DATA/unimed_benchmark

Tiles are given at our full resolution (512 px, about 0.5 um/px, the scale UNI and Virchow2 were
trained at); position embeddings are interpolated to the larger grid (timm dynamic_img_size,
DINOv3's rotary embeddings, bicubic resize for UniMed-CLIP).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_SERVER_BENCH = Path(__file__).resolve().parent.parent / "weights" / "unimed_benchmark"     # H100 server (check_unimedclip.py sets it up)
BENCH = Path(os.environ["QP_UNIMED"]) if os.environ.get("QP_UNIMED") else (
    _SERVER_BENCH if _SERVER_BENCH.exists() else Path(str(P.DATA_ROOT)) / "unimed_benchmark")


def even_blocks(n):
    return [n // 4, n // 2, 3 * n // 4, n]                      # 1-based block indices


class TimmViT(nn.Module):
    def __init__(self, hub, **kwargs):
        super().__init__()
        import timm
        self.vit = timm.create_model(f"hf-hub:{hub}", pretrained=True, dynamic_img_size=True, **kwargs)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.blocks = even_blocks(len(self.vit.blocks))
        self.dims = [self.vit.embed_dim] * 4

    def forward(self, x):
        return self.vit.forward_intermediates(x, indices=[b - 1 for b in self.blocks], norm=False,
                                              output_fmt="NCHW", intermediates_only=True)


class TimmFromCheckpoint(nn.Module):
    """A timm ViT built from an architecture name and filled from a checkpoint's sub-dict
    (CONCH: the vision trunk of its CoCa model, keys visual.trunk.*), loaded strictly."""

    def __init__(self, arch, hub, filename, prefix, **kwargs):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download
        self.vit = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True, **kwargs)
        sd = torch.load(hf_hub_download(hub, filename), map_location="cpu", weights_only=False)
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        self.vit.load_state_dict(sd, strict=True)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.blocks = even_blocks(len(self.vit.blocks))
        self.dims = [self.vit.embed_dim] * 4

    forward = TimmViT.forward


class DinoV3(nn.Module):
    def __init__(self, hub):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(hub)
        cfg = self.vit.config
        self.patch = cfg.patch_size
        self.skip = 1 + getattr(cfg, "num_register_tokens", 0)
        self.blocks = even_blocks(cfg.num_hidden_layers)
        self.dims = [cfg.hidden_size] * 4

    def forward(self, x):
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        hidden = self.vit(pixel_values=x, output_hidden_states=True).hidden_states   # [embeddings, block 1, ...]
        return [hidden[b][:, self.skip:].transpose(1, 2).reshape(x.shape[0], -1, h, w) for b in self.blocks]


def extra_wheels():
    """Pure-python wheels placed in weights/ (e.g. ftfy, wcwidth) are importable straight from the zip: nothing is installed in the venv."""
    for w in sorted((Path(__file__).resolve().parent.parent / "weights").glob("*.whl")):
        if str(w) not in sys.path:
            sys.path.append(str(w))


class UniMedCLIP(nn.Module):
    def __init__(self, ckpt="checkpoints/unimed_clip_vit_l14_base_text_encoder.pt", arch="ViT-L-14-336-quickgelu",
                 text="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"):
        super().__init__()
        extra_wheels()
        sys.path.insert(0, str(BENCH))
        from bench.encoders import load_unimed_clip
        from bench.vit import run_blocks
        self._run_blocks = run_blocks
        model, self.mean, self.std = load_unimed_clip(BENCH / "UniMed-CLIP", BENCH / ckpt, arch, text)
        self.visual = model.visual
        del model
        self.patch = self.visual.conv1.kernel_size[0]
        n = len(self.visual.transformer.resblocks)
        self.blocks = even_blocks(n)
        self.dims = [self.visual.conv1.out_channels] * 4

    def forward(self, x):
        v = self.visual
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        t = v.conv1(x).flatten(2).transpose(1, 2)
        pos = v.positional_embedding
        g = int(round((pos.shape[0] - 1) ** 0.5))
        grid = F.interpolate(pos[1:].reshape(1, g, g, -1).permute(0, 3, 1, 2).float(), size=(h, w),
                             mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(h * w, -1)
        cls = v.class_embedding.expand(x.shape[0], 1, -1)
        t = torch.cat([cls, t], 1) + torch.cat([pos[:1], grid.to(pos.dtype)], 0).to(t.dtype)
        t = v.ln_pre(t)
        out = []
        for i, blk in enumerate(v.transformer.resblocks, 1):
            t = self._run_blocks([blk], t)
            if i in self.blocks:
                out.append(t[:, 1:].transpose(1, 2).reshape(x.shape[0], -1, h, w))
        return out


def build_encoder(name):
    if name == "uni":
        enc = TimmViT("MahmoodLab/UNI", init_values=1e-5)
        enc.mean, enc.std = IMAGENET
    elif name == "virchow2":
        import timm
        enc = TimmViT("paige-ai/Virchow2", mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        enc.mean, enc.std = IMAGENET
    elif name == "dinov3":
        enc = DinoV3("facebook/dinov3-vitl16-pretrain-lvd1689m")
        enc.mean, enc.std = IMAGENET
    elif name == "dinov2":                                   # same wrapper: HF hidden states, no register tokens
        enc = DinoV3("facebook/dinov2-large")
        enc.mean, enc.std = IMAGENET
    elif name == "conch":
        enc = TimmFromCheckpoint("vit_base_patch16_224", "MahmoodLab/CONCH", "pytorch_model.bin", "visual.trunk.", img_size=448)
        enc.mean, enc.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)   # CONCH = OpenAI CLIP values
    elif name == "unimedclip":
        enc = UniMedCLIP()
    else:
        raise ValueError(f"unknown encoder {name}")
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval()


ENCODERS = ["uni", "virchow2", "conch", "dinov2", "dinov3", "unimedclip"]
