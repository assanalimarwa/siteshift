"""U-Net models: one input RGB tile -> one logit per pixel (tumour)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = conv_block(cin + cskip, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-34 encoder (the layout of
    segmentation_models_pytorch's Unet("resnet34")). Converges much faster than a U-Net
    trained from scratch."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)            # 64 ch, 1/2
        self.pool = r.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = r.layer1, r.layer2, r.layer3, r.layer4  # 1/4 .. 1/32
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        return self.head(self.dec0(d))


class UNet(nn.Module):
    """The original U-Net (Ronneberger et al. 2015) with padded convs and batch norm,
    trained from scratch."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        ch = [width * 2 ** i for i in range(depth + 1)]               # 64 .. 1024
        self.down = nn.ModuleList([conv_block(3, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()                                                   # the bottleneck has no skip
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


class _Shared:
    """Holds the frozen encoder outside the module tree: it is not a parameter, not in
    state_dict (checkpoints keep only the head) and not copied by deepcopy (the weight EMA
    shares it instead of duplicating up to 630 M frozen weights)."""

    def __init__(self, module):
        self.module = module

    def __deepcopy__(self, memo):
        return self


class FeatureHead(nn.Module):
    """The same head for every encoder: four block outputs -> 1x1 projection to 256 channels
    each -> concatenated and fused by two 3x3 convs -> three x2 upsampling conv blocks -> logit."""

    def __init__(self, dims, width=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(d, width, 1, bias=False), nn.BatchNorm2d(width),
                                                 nn.ReLU(inplace=True)) for d in dims])
        self.fuse = conv_block(len(dims) * width, width)
        self.up = nn.Sequential(DecoderBlock(width, 0, 128), DecoderBlock(128, 0, 64), DecoderBlock(64, 0, 32))
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, feats, size):
        x = self.fuse(torch.cat([p(f) for p, f in zip(self.proj, feats)], 1))
        return F.interpolate(self.out(self.up(x)), size=size, mode="bilinear", align_corners=False)


class FrozenEncoderSeg(nn.Module):
    """A frozen foundation-model encoder (encoders.py) with the trainable FeatureHead.

    Takes the pipeline's ImageNet-normalised input, converts it to the encoder's own
    normalisation, resizes it to the nearest multiple of the patch size (512 -> 518 for
    patch 14), and upsamples the logits back to the input size."""

    def __init__(self, name):
        super().__init__()
        from encoders import build_encoder
        enc = build_encoder(name)
        self.enc = _Shared(enc)
        self.patch = enc.patch
        self.head = FeatureHead(enc.dims)
        mean_in, std_in = torch.tensor((0.485, 0.456, 0.406)), torch.tensor((0.229, 0.224, 0.225))
        mean, std = torch.tensor(enc.mean), torch.tensor(enc.std)
        self.register_buffer("scale", (std_in / std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((mean_in - mean) / std).view(1, 3, 1, 1), persistent=False)

    def _apply(self, fn, *args, **kwargs):
        self.enc.module._apply(fn, *args, **kwargs)
        return super()._apply(fn, *args, **kwargs)

    def forward(self, x):
        size = x.shape[-2:]
        x = x * self.scale + self.shift                      # ImageNet-normalised -> encoder-normalised
        side = [max(self.patch, round(s / self.patch) * self.patch) for s in size]
        if list(size) != side:
            x = F.interpolate(x, size=side, mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self.enc.module(x.contiguous())
        return self.head(feats, size)


class SegFormerB2(nn.Module):
    """SegFormer-B2 (MiT-B2 encoder pretrained on ImageNet-1k, nvidia/mit-b2; new all-MLP decode head, 1 logit);
    the logits come at 1/4 size and are upsampled to the input size."""

    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=1)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained("nvidia/mit-b2", num_labels=1))

    def forward(self, x):
        return F.interpolate(self.net(pixel_values=x).logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class TransUNet(nn.Module):
    """TransUNet (Chen et al. 2021): ResNet-50 + ViT-B/16 hybrid encoder (timm vit_base_r50_s16_224.orig_in21k, ImageNet-21k,
    position embeddings interpolated to 512 px) and the cascaded-upsampler decoder with skips from the ResNet stages
    at 1/2, 1/4 and 1/8 (the original layout: decoder channels 256, 128, 64, 16)."""

    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        vit = timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=pretrained, img_size=512, num_classes=0)
        self.vit = vit
        bb = vit.patch_embed.backbone
        self.stem, self.stages = bb.stem, bb.stages                       # 64 ch 1/2 | 256 1/4, 512 1/8, 1024 1/16
        self.neck = nn.Sequential(nn.Conv2d(768, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.dec1 = DecoderBlock(512, 512, 256)                           # 1/16 -> 1/8, skip 512
        self.dec2 = DecoderBlock(256, 256, 128)                           # -> 1/4, skip 256
        self.dec3 = DecoderBlock(128, 64, 64)                             # -> 1/2, skip 64
        self.dec4 = DecoderBlock(64, 0, 16)                               # -> 1/1
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        s0 = self.stem.norm(self.stem.conv(x))                            # 64 ch at 1/2, before the max-pool
        s1 = self.stages[0](self.stem.pool(s0))
        s2 = self.stages[1](s1)
        f = self.stages[2](s2)
        b, c, h, w = f.shape
        t = self.vit.patch_embed.proj(f).flatten(2).transpose(1, 2)       # tokens at 1/16
        t = self.vit._pos_embed(t)
        t = self.vit.norm(self.vit.blocks(self.vit.norm_pre(t)))
        t = t[:, self.vit.num_prefix_tokens:].transpose(1, 2).reshape(b, -1, h, w)
        d = self.neck(t)
        d = self.dec1(d, s2)
        d = self.dec2(d, s1)
        d = self.dec3(d, s0)
        return self.head(self.dec4(d))


class NNUNetPlain(nn.Module):
    """The 2D U-Net nnU-Net plans for 512-px inputs (PlainConvUNet): 7 stages with 32, 64, 128, 256, 512, 512, 512
    features, two 3x3 convs per stage, instance norm, leaky ReLU, strided-conv downsampling, transposed-conv upsampling,
    random initialisation, and deep supervision (a 1x1 output on every decoder stage; train mode returns them all,
    full size first). One logit per pixel (binary tumour), as the other models here."""

    FEATURES = (32, 64, 128, 256, 512, 512, 512)

    @staticmethod
    def block(cin, cout, stride):
        def cn(i, o, s):
            return [nn.Conv2d(i, o, 3, s, 1, bias=True), nn.InstanceNorm2d(o, eps=1e-5, affine=True), nn.LeakyReLU(0.01, inplace=True)]
        return nn.Sequential(*cn(cin, cout, stride), *cn(cout, cout, 1))

    def __init__(self):
        super().__init__()
        f = self.FEATURES
        self.enc = nn.ModuleList([self.block(3 if i == 0 else f[i - 1], f[i], 1 if i == 0 else 2) for i in range(len(f))])
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, 2) for i in range(len(f) - 1)])
        self.dec = nn.ModuleList([self.block(2 * f[i], f[i], 1) for i in range(len(f) - 1)])
        self.out = nn.ModuleList([nn.Conv2d(f[i], 1, 1) for i in range(len(f) - 1)])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        outs = []
        for i in reversed(range(len(self.up))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
            outs.append(self.out[i](x))
        outs = outs[::-1]                                                 # full size first
        return outs if self.training else outs[0]


def build_model(arch, pretrained=True):
    if arch == "resnet34":
        return ResNetUNet(pretrained)
    if arch == "unet":
        return UNet()
    if arch == "segformer":
        return SegFormerB2(pretrained)
    if arch == "transunet":
        return TransUNet(pretrained)
    if arch == "nnunet":
        return NNUNetPlain()
    if arch.startswith("fm_"):
        return FrozenEncoderSeg(arch[3:])                    # the encoder is always the published one
    raise ValueError(arch)
