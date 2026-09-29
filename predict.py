import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLAHE_CLIP, CLAHE_GRID = 1.0, (8, 8)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def load_config(config_path=DEFAULT_CONFIG):
    return json.loads(Path(config_path).read_text())


def minmax_uint8(arr):
    arr = np.asarray(arr).astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    return ((arr - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8)


def _as_dataset(dcm):
    import pydicom

    if isinstance(dcm, (str, bytes)) or hasattr(dcm, "__fspath__"):
        return pydicom.dcmread(dcm, force=True)
    return dcm


def dicom_to_image(dcm):
    import cv2

    ds = _as_dataset(dcm)
    raw = ds.pixel_array.astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a single grayscale X-ray, got pixel shape {raw.shape}")
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        raw = raw.max() - raw
    base = minmax_uint8(raw)
    eq = cv2.equalizeHist(base)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(base)
    return np.stack([eq, clahe, base], axis=-1), base.shape


def letterbox_topleft(img, size):
    import cv2

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    padded[:nh, :nw, :] = resized
    return padded, (nh, nw)


def pad_square_topleft(img, fill=0):
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    out = np.full((s, s, img.shape[2]), fill, dtype=img.dtype)
    out[:h, :w, :] = img
    return out


def to_input_resized(img3, size, mean, std, device, dtype):
    x = torch.from_numpy(np.ascontiguousarray(img3)).permute(2, 0, 1)[None].float()
    if tuple(x.shape[-2:]) != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    x = x / 255.0
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return x.to(device=device, dtype=dtype)


def to_input(img3, mean, std, device, dtype):
    x = torch.from_numpy(np.ascontiguousarray(img3)).float().permute(2, 0, 1) / 255.0
    x = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return x.unsqueeze(0).to(device=device, dtype=dtype)


def present_fallback_mask(prob, fraction):
    prob = np.asarray(prob, dtype=np.float32)
    if float(prob.max()) <= 0.0:
        return np.zeros(prob.shape, dtype=np.uint8)
    n = max(1, int(prob.size * fraction))
    mask = np.zeros(prob.size, np.uint8)
    mask[np.argpartition(prob.ravel(), -n)[-n:]] = 1
    return mask.reshape(prob.shape)


def _group_norm(num_channels):
    groups = next(g for g in (32, 16, 8, 4, 2, 1) if num_channels % g == 0)
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False), _group_norm(out_ch), nn.GELU())

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(in_ch, out_ch), ConvNormAct(out_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6), reduction=4):
        super().__init__()
        inter = max(in_channels // reduction, 32)
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool2d(ps), ConvNormAct(in_channels, inter, k=1)) for ps in pool_sizes]
        )
        self.bottleneck = ConvNormAct(in_channels + inter * len(pool_sizes), out_channels, k=3)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for stage in self.stages:
            feats.append(F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(feats, dim=1))


class ConvNextUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, target_mask_size, ppm_pool_sizes):
        super().__init__()
        self.target_mask_size = target_mask_size
        self.bottleneck = PPM(encoder_channels[0], decoder_channels[0], pool_sizes=ppm_pool_sizes)
        self.up_blocks = nn.ModuleList([
            DoubleConv(decoder_channels[i] + encoder_channels[i + 1], decoder_channels[i + 1])
            for i in range(len(encoder_channels) - 1)
        ])
        self.final_refine = DoubleConv(decoder_channels[-1], decoder_channels[-1])
        self.head = nn.Conv2d(decoder_channels[-1], out_channels, 1)

    def forward(self, feature_maps):
        x = self.bottleneck(feature_maps[0])
        for up, skip in zip(self.up_blocks, feature_maps[1:]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], dim=1))
        x = self.final_refine(x)
        x = F.interpolate(x, size=(self.target_mask_size, self.target_mask_size), mode="bilinear", align_corners=False)
        return self.head(x)


class ConvNextTower(nn.Module):
    def __init__(self, tower_cfg):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.for_model(tower_cfg["model_type"], **{k: v for k, v in tower_cfg.items() if k != "model_type"})
        cfg.output_hidden_states = True
        self.backbone = AutoModel.from_config(cfg)
        hs = tuple(getattr(cfg, "hidden_sizes", None) or (192, 384, 768, 1536))
        self.encoder_channels = tuple(reversed(hs[-4:]))

    def forward(self, pixel_values):
        maps = list(self.backbone(pixel_values).hidden_states[-4:])
        return list(reversed(maps))


def build_vit_tower(tower_cfg, patch_start):
    from transformers import AutoConfig, AutoModel

    expected = 1 + int(tower_cfg.get("num_register_tokens", 0))
    if int(patch_start) != expected:
        raise ValueError(f"patch_start {patch_start} != 1 CLS + {expected - 1} register tokens")
    cfg = AutoConfig.for_model(tower_cfg["model_type"], **{k: v for k, v in tower_cfg.items() if k != "model_type"})
    return AutoModel.from_config(cfg)


class SegModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.vision_model = ConvNextTower(spec["tower"])
        self.segmentation_decoder = ConvNextUNetDecoder(self.vision_model.encoder_channels, **spec["decoder"])

    def forward(self, pixel_values):
        return self.segmentation_decoder(self.vision_model(pixel_values))


class TokenPooler(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class ClassifierHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class ClsModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.patch_start = int(spec["patch_start"])
        self.vision_model = build_vit_tower(spec["tower"], self.patch_start)
        self.pool = TokenPooler(spec["dim"], **spec["pool"])
        self.head = ClassifierHead(spec["dim"], **spec["head"])

    def forward(self, pixel_values):
        tokens = self.vision_model(pixel_values).last_hidden_state[:, self.patch_start:, :]
        return self.head(self.pool(tokens))


class Task1Model:
    def __init__(self, cls_net, seg_nets, seg_spec, cls_spec):
        self.cls_net = cls_net
        self.seg_nets = seg_nets
        self.seg_size = int(seg_spec["infer_size"])
        self.cls_size = int(cls_spec["infer_size"])
        self.op = seg_spec["op"]
        param = next(cls_net.parameters())
        self.device, self.dtype = param.device, param.dtype

    @torch.no_grad()
    def _cls_prob(self, dcm):
        img3, _ = dicom_to_image(dcm)
        x = to_input_resized(pad_square_topleft(img3), self.cls_size, IMAGENET_MEAN, IMAGENET_STD,
                             self.device, self.dtype)
        return float(torch.sigmoid(self.cls_net(x).reshape(-1)[0].float()))

    @torch.no_grad()
    def _prob(self, dcm):
        img3, native_hw = dicom_to_image(dcm)
        padded, (nh, nw) = letterbox_topleft(img3, self.seg_size)
        x = to_input(padded, IMAGENET_MEAN, IMAGENET_STD, self.device, self.dtype)
        member_probs = [net(x).sigmoid()[0, 0] for net in self.seg_nets]
        prob = torch.stack(member_probs).mean(0)[:nh, :nw]
        prob = F.interpolate(prob[None, None].float(), size=native_hw, mode="bilinear", align_corners=False)[0, 0]
        return prob.float().cpu().numpy()

    @torch.no_grad()
    def predict(self, dcm_path):
        dcm = _as_dataset(dcm_path)
        present = int(self._cls_prob(dcm) >= self.op["detection_tau"])
        if not self.seg_nets:
            return None, present
        prob = self._prob(dcm)
        mask = (prob > self.op["threshold"]).astype(np.uint8) if present else np.zeros(prob.shape, np.uint8)
        if present and not mask.any():
            mask = present_fallback_mask(prob, self.op["fallback_fraction"])
        return mask, present


def load_model(weights_path=DEFAULT_WEIGHTS, mode="seg", device="auto", config_path=DEFAULT_CONFIG):
    from safetensors.torch import load_file

    if mode not in ("seg", "cls"):
        raise ValueError("mode must be 'seg' or 'cls'")
    weights_path = Path(weights_path)
    paths = [weights_path / f"model_{i}.safetensors" for i in range(5 if mode == "seg" else 1)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {path}")
    config = load_config(config_path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    nets = []
    for i, path in enumerate(paths):
        net = ClsModel(config["cls_spec"]) if i == 0 else SegModel(config["seg_spec"])
        net.load_state_dict(load_file(str(path)), strict=True)
        nets.append(net.to(dev).eval())
        print(f"[weights] loaded {path.name}", flush=True)
    return Task1Model(nets[0], nets[1:], config["seg_spec"], config["cls_spec"])


def dicom_files(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".dcm", ".dicom"))


def discover_cases(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in (".dcm", ".dicom"):
            raise ValueError(f"Expected a .dcm or .dicom file: {path}")
        return [(path.stem, path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = dicom_files(path)
    case_dirs = sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    if files and case_dirs:
        raise ValueError("Input must contain either DICOM files or case folders, not both")
    cases = [(p.stem, p) for p in files]
    for directory in case_dirs:
        paths = dicom_files(directory)
        if not paths:
            raise ValueError(f"No DICOM files found in case folder: {directory}")
        cases.append((directory.name, paths[0]))
    if not cases:
        raise ValueError(f"No DICOM inputs found in: {path}")
    ids = [case_id for case_id, _ in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Input files have duplicate case IDs")
    return cases


def make_mask_match_reference(mask, reference):
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8)

    ref_dim = reference.GetDimension()
    ref_size = reference.GetSize()

    if ref_dim == 2:
        expected_shape = (ref_size[1], ref_size[0])
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise RuntimeError(f"Expected 2D mask for 2D X-ray reference, got shape {mask.shape}")
        if mask.shape != expected_shape:
            raise RuntimeError(f"Mask shape {mask.shape} does not match reference shape {expected_shape}")
    elif ref_dim == 3:
        expected_shape = (ref_size[2], ref_size[1], ref_size[0])
        if mask.ndim == 2 and ref_size[2] == 1:
            mask = mask[None, :, :]
        if mask.ndim != 3:
            raise RuntimeError(f"Expected 3D mask for 3D reference, got shape {mask.shape}")
        if mask.shape != expected_shape:
            raise RuntimeError(f"Mask shape {mask.shape} does not match reference shape {expected_shape}")
    else:
        raise RuntimeError(f"Unsupported reference dimension: {ref_dim}")

    return mask.astype(np.uint8)


def save_mask(mask, dcm_path, output_path):
    import SimpleITK as sitk

    reference = sitk.ReadImage(str(dcm_path))
    mask = make_mask_match_reference(mask, reference)
    pred = sitk.GetImageFromArray(mask)
    pred.CopyInformation(reference)
    sitk.WriteImage(pred, str(output_path))


def run_one_case(model, case_id, dcm_path, output_dir):
    mask, cavity = model.predict(dcm_path)
    if mask is not None:
        save_mask(mask, dcm_path, Path(output_dir) / f"{case_id}.nii.gz")
    return case_id, cavity


def predict(input_path, output_dir, weights_path=DEFAULT_WEIGHTS, mode="seg", device="auto", config_path=DEFAULT_CONFIG):
    cases = discover_cases(input_path)
    model = load_model(weights_path, mode=mode, device=device, config_path=config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one_case(model, case_id, path, output_dir) for case_id, path in cases]
    with (output_dir / "prediction.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["our_id", "cavity"])
        writer.writerows(rows)
    print(f"[done] wrote {len(rows)} predictions (mode={mode}) -> {output_dir}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="MultimodalAI: cavity detection and segmentation")
    parser.add_argument("--input", default="/input", help="DICOM file, flat DICOM folder, or root of case folders")
    parser.add_argument("--output", default="/output", help="output directory")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="directory containing model_0.safetensors through model_4.safetensors")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.json holding model and threshold settings")
    parser.add_argument("--mode", choices=("seg", "cls"), default="seg", help="seg: classification and masks (default); cls: classification only")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    predict(args.input, args.output, args.weights, args.mode, args.device, args.config)


if __name__ == "__main__":
    main()
