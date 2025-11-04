"""Core metric computations for the stg_light_eval toolkit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Optional torch/timm dependencies.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - functions fall back to numpy ops.
    torch = None  # type: ignore
    F = None  # type: ignore

try:  # Backbones pulled via timm when available.
    import timm
except ImportError:  # pragma: no cover
    timm = None  # type: ignore

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow should exist but guard anyway.
    Image = None  # type: ignore


__all__ = [
    "rule_f1",
    "tree_edit_distance",
    "repeat_regularity_error",
    "mdl_score",
    "persistence_score",
    "motion_error",
    "feature_recon_cosine",
    "depth_rmse",
    "normal_agreement",
]


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if Image is not None and isinstance(value, Image.Image):
        return np.array(value)
    return np.asarray(value)


def _match_rules(gt_rules: Sequence[Mapping[str, Any]], pred_rules: Sequence[Mapping[str, Any]], tol: Mapping[str, float]) -> int:
    matched = 0
    gt_used = [False] * len(gt_rules)

    for pred in pred_rules:
        best_idx = -1
        best_score = float("inf")
        for idx, gt_rule in enumerate(gt_rules):
            if gt_used[idx]:
                continue
            score = 0.0
            compatible = True
            for key, gt_val in gt_rule.items():
                if key not in pred:
                    compatible = False
                    break
                pred_val = pred[key]
                if isinstance(gt_val, (int, float, np.number)) and isinstance(pred_val, (int, float, np.number)):
                    allowed = tol.get(key, 0.0)
                    if abs(float(gt_val) - float(pred_val)) > allowed:
                        compatible = False
                        break
                    score += abs(float(gt_val) - float(pred_val))
                else:
                    gt_arr = _to_numpy(gt_val)
                    pred_arr = _to_numpy(pred_val)
                    if gt_arr.shape != pred_arr.shape:
                        compatible = False
                        break
                    allowed = tol.get(key, 0.0)
                    diff = float(np.linalg.norm(gt_arr.astype(np.float64) - pred_arr.astype(np.float64)))
                    if diff > allowed:
                        compatible = False
                        break
                    score += diff
            if compatible and score < best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            gt_used[best_idx] = True
            matched += 1
    return matched


def rule_f1(
    gt_rules: Iterable[Mapping[str, Any]],
    pred_rules: Iterable[Mapping[str, Any]],
    tol: Mapping[str, float] | None = None,
) -> tuple[float, float, float]:
    tol = dict(tol or {"pos": 0.03, "count": 1.0})
    gt_list = list(gt_rules)
    pred_list = list(pred_rules)
    if not gt_list and not pred_list:
        return 1.0, 1.0, 1.0
    if not gt_list:
        return 0.0, 1.0, 0.0
    if not pred_list:
        return 1.0, 0.0, 0.0
    matched = _match_rules(gt_list, pred_list, tol)
    precision = matched / len(pred_list) if pred_list else 0.0
    recall = matched / len(gt_list) if gt_list else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _tree_children(node: Any) -> Sequence[Any]:
    if isinstance(node, Mapping):
        children = node.get("children")
        if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
            return children
        return []
    if isinstance(node, (list, tuple)):
        return node
    return []


def _tree_label(node: Any) -> str:
    if isinstance(node, Mapping):
        for key in ("symbol", "label", "name", "id"):
            if key in node:
                return str(node[key])
    if isinstance(node, str):
        return node
    return str(node)


def _tree_size(node: Any) -> int:
    children = _tree_children(node)
    return 1 + sum(_tree_size(child) for child in children)


def _tree_edit_cost(a: Any, b: Any) -> int:
    label_cost = 0 if _tree_label(a) == _tree_label(b) else 1
    children_a = list(_tree_children(a))
    children_b = list(_tree_children(b))
    if not children_a and not children_b:
        return label_cost

    size_cache = {}

    def subtree_size(node: Any) -> int:
        key = id(node)
        if key not in size_cache:
            size_cache[key] = _tree_size(node)
        return size_cache[key]

    dp = np.zeros((len(children_a) + 1, len(children_b) + 1), dtype=np.int32)
    for i in range(1, len(children_a) + 1):
        dp[i, 0] = dp[i - 1, 0] + subtree_size(children_a[i - 1])
    for j in range(1, len(children_b) + 1):
        dp[0, j] = dp[0, j - 1] + subtree_size(children_b[j - 1])

    for i in range(1, len(children_a) + 1):
        for j in range(1, len(children_b) + 1):
            cost_sub = _tree_edit_cost(children_a[i - 1], children_b[j - 1])
            dp[i, j] = min(
                dp[i - 1, j] + subtree_size(children_a[i - 1]),  # deletion
                dp[i, j - 1] + subtree_size(children_b[j - 1]),  # insertion
                dp[i - 1, j - 1] + cost_sub,  # substitution
            )
    return int(label_cost + dp[-1, -1])


def tree_edit_distance(gt_tree: Any, pred_tree: Any) -> float:
    size_gt = _tree_size(gt_tree)
    size_pred = _tree_size(pred_tree)
    denom = max(size_gt, size_pred, 1)
    cost = _tree_edit_cost(gt_tree, pred_tree)
    return min(cost / denom, 1.0)


def repeat_regularity_error(pred_nodes: Iterable[Any]) -> float:
    nodes = list(pred_nodes)
    if len(nodes) < 3:
        return 0.0
    pts = []
    for node in nodes:
        if isinstance(node, Mapping):
            coord = None
            for key in ("pos", "position", "center"):
                if key in node:
                    coord = _to_numpy(node[key])
                    break
            if coord is not None:
                pts.append(coord.astype(np.float32))
        else:
            pts.append(_to_numpy(node).astype(np.float32))
    if not pts:
        return 0.0
    pts_np = np.stack([p if p.ndim == 1 else p.flatten() for p in pts], axis=0)
    if pts_np.shape[1] > 2:
        pts_np = pts_np[:, :2]
    axis = int(np.argmax(np.var(pts_np, axis=0)))
    sorted_vals = np.sort(pts_np[:, axis])
    spacings = np.diff(sorted_vals)
    if spacings.size == 0:
        return 0.0
    return float(np.std(spacings))


def mdl_score(structure_cost_bits: Any, residual_cost_bits: Any, cap_bits: float = 1e6) -> float:
    total = max(float(structure_cost_bits) + float(residual_cost_bits), 0.0)
    cap = max(cap_bits, 1.0)
    return float(min(total / cap, 1.0))


def persistence_score(track_ids: Mapping[Any, Sequence[Any]] | Sequence[Sequence[Any]]) -> float:
    if isinstance(track_ids, Mapping):
        sequences = list(track_ids.values())
    else:
        sequences = list(track_ids)
    if not sequences:
        return 0.0
    scores = []
    for seq in sequences:
        items = [item for item in seq if item is not None]
        if not items:
            scores.append(0.0)
            continue
        counts = Counter(items)
        dominant = counts.most_common(1)[0][1]
        scores.append(dominant / len(seq))
    return float(np.mean(scores))


def motion_error(gt_params: Any, pred_params: Any) -> float:
    gt = _to_numpy(gt_params).astype(np.float64)
    pred = _to_numpy(pred_params).astype(np.float64)
    diff = float(np.linalg.norm(gt - pred))
    value_range = float(np.max(gt) - np.min(gt)) if gt.size else 0.0
    if value_range <= 1e-8:
        return 0.0 if diff <= 1e-8 else 1.0
    return float(min(diff / value_range, 1.0))


_BACKBONES: dict[str, Any] = {}


def _prepare_image_tensor(img: Any, size: int = 224) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("torch is required for feature-based cosine similarity")
    if Image is not None and isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        arr = _to_numpy(img)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        if Image is None:
            raise RuntimeError("pillow is required for image preprocessing")
        pil = Image.fromarray(arr)
    pil = pil.resize((size, size))
    arr = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _load_backbone(name: str) -> Any:
    if name in _BACKBONES:
        return _BACKBONES[name]
    if timm is None or torch is None:
        raise RuntimeError("timm and torch are required for backbone features")
    if name == "dino_v2":
        model_name = "vit_small_patch16_224.dino.v2"
    else:
        model_name = name
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    _BACKBONES[name] = model
    return model


def _extract_features(img: Any, backbone: str) -> np.ndarray:
    if torch is None or timm is None:
        arr = _to_numpy(img).astype(np.float32)
        flat = arr.flatten()
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 1e-8 else flat
    model = _load_backbone(backbone)
    tensor = _prepare_image_tensor(img)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        feats = model.forward_features(tensor)
        if isinstance(feats, Mapping):
            if "x_norm_clstoken" in feats:
                feats = feats["x_norm_clstoken"]
            else:
                feats = next(iter(feats.values()))
        if hasattr(model, "forward_head"):
            feats = model.forward_head(feats, pre_logits=True)
        if feats.ndim > 2:
            feats = feats.mean(dim=(-2, -1))
        feats = feats.flatten(1)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()[0]


def feature_recon_cosine(img: Any, recon_img: Any, backbone: str = "dino_v2") -> float:
    feat_a = _extract_features(img, backbone)
    feat_b = _extract_features(recon_img, backbone)
    denom = float(np.linalg.norm(feat_a) * np.linalg.norm(feat_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(feat_a, feat_b) / denom)


def depth_rmse(pred_depth: Any, gt_depth: Any) -> float:
    pred = _to_numpy(pred_depth).astype(np.float64)
    gt = _to_numpy(gt_depth).astype(np.float64)
    diff = pred - gt
    return float(np.sqrt(np.mean(diff ** 2)))


def normal_agreement(pred_normals: Any, gt_normals: Any) -> float:
    pred = _to_numpy(pred_normals).astype(np.float64)
    gt = _to_numpy(gt_normals).astype(np.float64)
    pred = pred.reshape(-1, pred.shape[-1])
    gt = gt.reshape(-1, gt.shape[-1])
    pred_norm = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None)
    gt_norm = gt / np.clip(np.linalg.norm(gt, axis=1, keepdims=True), 1e-8, None)
    cos = np.sum(pred_norm * gt_norm, axis=1)
    return float(np.mean((cos + 1.0) * 0.5))
