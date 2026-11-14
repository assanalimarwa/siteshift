"""
UNI embeddings for fig_domains.pdf (GPU, a few minutes):

    python paper_figs/src/embed_uni.py      (from the unet folder)

Up to 2,000 random tissue tiles (tissue share >= 0.5, seed 0) per group: ours biopsies, ours large sections,
SICAPv2, PANDA-Karolinska, PANDA-Radboud (the tile sets prepared for the experiments). Every tile is brought to the
same field of about 256 um before UNI: ours / Karolinska / Radboud tiles (0.49-0.50 um/px) whole, SICAPv2 tiles
(about 1.75x smaller in px than ours) as their central 293 px; then resized to 224 px. Feature = UNI's class token.
Writes results/datasets/data/uni_embeddings.npz (not released) (feat, group, image).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])] + [str(_Path(__file__).resolve().parents[1] / _d) for _d in ('data', 'train', 'eval', 'figures')]
import siteshift_paths as P

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

U = P.INDEX
sys.path.insert(0, str(U))
from encoders import build_encoder  # noqa: E402

N, SEED, MIN_TISSUE, SICAP_ZOOM = 2000, 0, 0.5, 1.75
OUT = P.RESULTS / "datasets" / "data" / "uni_embeddings.npz"


def groups():
    info = pd.read_csv(P.RESULTS / "cv47" / "dataset_by_patient.csv").set_index("p").type
    t = pd.read_csv(U / "tiles47.csv")
    t = t[t.tissue_frac >= MIN_TISSUE]
    typ = t.slide.str.split(".").str[0].astype(int).map(info)
    d47 = P.DATASET47
    out = {"Ours, biopsies": (t[typ == "biopsy"], d47, 1.0), "Ours, large sections": (t[typ == "large"], d47, 1.0)}
    for name, idx, root, zoom in [("SICAPv2", "sicap_tiles.csv", str(P.DATA_ROOT / "sicapv2"), SICAP_ZOOM),
                                  ("PANDA-Karolinska", "karolinska_all_tiles.csv", str(P.DATA_ROOT / "karolinska_all"), 1.0),
                                  ("PANDA-Radboud", "radboud_tiles.csv", str(P.DATA_ROOT / "radboud"), 1.0)]:
        x = pd.read_csv(U / idx)
        out[name] = (x[x.tissue_frac >= MIN_TISSUE], Path(root), zoom)
    return {k: (v.sample(min(N, len(v)), random_state=SEED), r, z) for k, (v, r, z) in out.items()}


@torch.no_grad()
def main():
    enc = build_encoder("uni").cuda().eval()
    mean = torch.tensor(enc.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(enc.std, device="cuda").view(1, 3, 1, 1)
    feats, labels, images = [], [], []
    for name, (rows, root, zoom) in groups().items():
        batch = []
        for i, rel in enumerate(rows.image):
            im = Image.open(root / rel).convert("RGB")
            if zoom != 1.0:
                side = round(im.width / zoom)
                o = (im.width - side) // 2
                im = im.crop((o, o, o + side, o + side))
            batch.append(np.asarray(im.resize((224, 224), Image.BICUBIC)))
            if len(batch) == 64 or i == len(rows) - 1:
                x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = enc.vit.forward_head(enc.vit.forward_features((x - mean) / std), pre_logits=True)
                feats.append(f.float().cpu().numpy())
                batch = []
        labels += [name] * len(rows)
        images += list(rows.image)
        print(f"{name}: {len(rows)} tiles", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, feat=np.concatenate(feats), group=np.array(labels), image=np.array(images))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
