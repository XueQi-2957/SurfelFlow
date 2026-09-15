"""Metric-only geometry evaluation for registered RGB-D sequences.

The evaluator is deliberately independent from the training code.  It reads
raw expected-depth/opacity arrays exported after a frame has been optimized,
matches them to a TUM/Bonn depth stream, and reports scale-aligned depth and
depth-derived normal metrics.  No CUDA or model import is required.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a project dependency
    Image = None


BONN_INTRINSICS = {
    "fx": 542.822841,
    "fy": 542.576870,
    "cx": 315.593520,
    "cy": 237.756098,
}
BONN_IMAGE_SIZE = (640, 480)  # width, height
GEOMETRY_MANIFEST_FORMAT = "gflow-geometry-v1"


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_value(cls, value: "Intrinsics | Mapping[str, float] | Sequence[float]") -> "Intrinsics":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                float(value["fx"]),
                float(value["fy"]),
                float(value["cx"]),
                float(value["cy"]),
            )
        if len(value) != 4:
            raise ValueError(f"Expected four intrinsics, got {value!r}")
        return cls(*(float(item) for item in value))

    def scaled(self, width: int, height: int, source_size: tuple[int, int] = BONN_IMAGE_SIZE) -> "Intrinsics":
        source_width, source_height = source_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid source image size: {source_size}")
        sx = float(width) / source_width
        sy = float(height) / source_height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)


@dataclass(frozen=True)
class PredictionFrame:
    frame_index: int
    frame_name: str
    depth_path: Path
    alpha_path: Path | None


def _numeric_key(path: Path | str) -> tuple[int, str]:
    match = re.findall(r"\d+", Path(path).stem)
    return (int(match[-1]) if match else 10**18, Path(path).name)


def _read_array(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            key = next(
                (candidate for candidate in ("depth_expected", "depth", "alpha", "arr_0") if candidate in archive),
                None,
            )
            if key is None:
                if not archive.files:
                    raise ValueError(f"NPZ file has no arrays: {path}")
                key = archive.files[0]
            value = archive[key]
        finally:
            archive.close()
    else:
        if Image is None:
            raise ImportError("Pillow is required to read PNG/TUM depth files")
        with Image.open(path) as image:
            value = np.asarray(image)

    value = np.asarray(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    while value.ndim > 2 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == 3:
        # A depth image should be single-channel.  Accept RGB files only when
        # all channels are equal; silently taking one channel can hide a bad
        # export.
        if value.shape[-1] in (3, 4) and np.all(value[..., 1:] == value[..., :1]):
            value = value[..., 0]
        else:
            raise ValueError(f"Expected a 2-D array at {path}, got shape {value.shape}")
    if value.ndim != 2:
        raise ValueError(f"Expected a 2-D array at {path}, got shape {value.shape}")
    return value


def load_depth(path: Path | str, png_scale: float = 5000.0, npy_scale: float = 1.0) -> np.ndarray:
    """Load metric depth in metres.

    Standard TUM/Bonn uint16 PNG depth uses ``value / 5000`` metres.  NPY
    predictions are assumed to already be in the renderer's depth units and
    are multiplied by ``npy_scale`` (one by default).
    """

    path = Path(path)
    value = _read_array(path)
    is_integer_image = path.suffix.lower() in {".png", ".tif", ".tiff", ".pgm"} and np.issubdtype(value.dtype, np.integer)
    if is_integer_image:
        scale = float(png_scale)
    else:
        scale = float(npy_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Depth scale must be positive and finite, got {scale}")
    value = value.astype(np.float32, copy=False)
    return value / scale if is_integer_image else value * scale


def load_alpha(path: Path | str) -> np.ndarray:
    """Load an opacity map and normalize integer images to ``[0, 1]``."""

    value = _read_array(Path(path))
    if np.issubdtype(value.dtype, np.integer) and value.size and int(value.max()) > 1:
        value = value.astype(np.float32) / 255.0
    return value.astype(np.float32, copy=False)


def resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array without interpolating invalid depth zeros."""

    array = np.asarray(array)
    if array.shape == shape:
        return array
    if array.ndim != 2 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Cannot resize shape {array.shape} to {shape}")
    out_h, out_w = shape
    in_h, in_w = array.shape
    y = np.floor((np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h).astype(np.int64)
    x = np.floor((np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w).astype(np.int64)
    y = np.clip(y, 0, in_h - 1)
    x = np.clip(x, 0, in_w - 1)
    return array[np.ix_(y, x)]


def _valid_depth(depth: np.ndarray, min_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > float(min_depth))


def depth_to_normals(depth: np.ndarray, intrinsics: Intrinsics, min_depth: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-z depth to central-difference camera-space normals."""

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got {depth.shape}")
    height, width = depth.shape
    normals = np.full((max(height - 2, 0), max(width - 2, 0), 3), np.nan, dtype=np.float32)
    valid = np.zeros(normals.shape[:2], dtype=bool)
    if height < 3 or width < 3:
        return normals, valid

    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.stack(
        [
            (xx - intrinsics.cx) * depth / intrinsics.fx,
            (yy - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        ],
        axis=-1,
    )
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    cross = np.cross(dx, dy)
    lengths = np.linalg.norm(cross, axis=-1)
    depth_valid = _valid_depth(depth, min_depth)
    valid = (
        depth_valid[1:-1, 1:-1]
        & depth_valid[:-2, 1:-1]
        & depth_valid[2:, 1:-1]
        & depth_valid[1:-1, :-2]
        & depth_valid[1:-1, 2:]
        & np.isfinite(lengths)
        & (lengths > 1e-8)
    )
    normals[valid] = cross[valid] / lengths[valid, None]
    return normals, valid


def _interior_all(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return np.zeros((max(mask.shape[0] - 2, 0), max(mask.shape[1] - 2, 0)), dtype=bool)
    return (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )


def _prepare_pair(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    alpha: np.ndarray | None,
    alpha_min: float | None,
    min_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2:
        raise ValueError(f"Expected prediction shape (H,W), got {prediction.shape}")
    target_shape = prediction.shape
    ground_truth = resize_nearest(np.asarray(ground_truth, dtype=np.float32), target_shape)
    gt_valid = _valid_depth(ground_truth, min_depth)
    pred_valid = _valid_depth(prediction, min_depth)
    if alpha is None:
        alpha_valid = np.ones(target_shape, dtype=bool)
    else:
        alpha = resize_nearest(np.asarray(alpha, dtype=np.float32), target_shape)
        alpha_valid = np.isfinite(alpha) & (alpha >= float(alpha_min if alpha_min is not None else 0.0))
    common = gt_valid & pred_valid & alpha_valid
    return prediction, ground_truth, gt_valid, common


def estimate_scale(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray, min_pixels: int = 100) -> float:
    """Estimate one multiplicative prediction-to-ground-truth scale."""

    count = int(valid.sum())
    if count < int(min_pixels):
        raise ValueError(f"Only {count} valid pixels are available for scale estimation; need {min_pixels}")
    ratios = ground_truth[valid] / prediction[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < int(min_pixels):
        raise ValueError(f"Only {ratios.size} finite scale ratios are available; need {min_pixels}")
    scale = float(np.median(ratios))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Estimated invalid depth scale: {scale}")
    return scale


def compute_frame_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    scale: float | None = None,
    intrinsics: Intrinsics | Mapping[str, float] | Sequence[float] = BONN_INTRINSICS,
    intrinsics_size: tuple[int, int] = BONN_IMAGE_SIZE,
    alpha_min: float | None = 0.01,
    min_depth: float = 1e-6,
    min_scale_pixels: int = 100,
) -> dict[str, Any]:
    """Compute scale-aligned depth and depth-derived normal metrics for a frame."""

    prediction, ground_truth, gt_valid, common = _prepare_pair(
        prediction, ground_truth, alpha, alpha_min, min_depth
    )
    if scale is None:
        scale = estimate_scale(prediction, ground_truth, common, min_scale_pixels)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Scale must be positive and finite, got {scale}")

    aligned = prediction * scale
    gt_count = int(gt_valid.sum())
    common_count = int(common.sum())
    result: dict[str, Any] = {
        "status": "ok" if common_count else "no_valid_pixels",
        "scale": scale,
        "gt_valid_pixels": gt_count,
        "common_valid_pixels": common_count,
        "coverage": float(common_count / gt_count) if gt_count else math.nan,
        "abs_rel": math.nan,
        "rmse": math.nan,
        "delta1": math.nan,
        "normal_mae_deg": math.nan,
        "normal_valid_pixels": 0,
        "normal_coverage": math.nan,
    }
    if common_count == 0:
        return result

    pred_values = aligned[common]
    gt_values = ground_truth[common]
    result["abs_rel"] = float(np.mean(np.abs(pred_values - gt_values) / gt_values))
    result["rmse"] = float(np.sqrt(np.mean(np.square(pred_values - gt_values))))
    ratio = np.maximum(pred_values / gt_values, gt_values / pred_values)
    result["delta1"] = float(np.mean(ratio < 1.25))

    intrinsics = Intrinsics.from_value(intrinsics).scaled(
        prediction.shape[1], prediction.shape[0], source_size=intrinsics_size
    )
    pred_normals, pred_normal_valid = depth_to_normals(aligned, intrinsics, min_depth)
    gt_normals, gt_normal_valid = depth_to_normals(ground_truth, intrinsics, min_depth)
    normal_valid = pred_normal_valid & gt_normal_valid & _interior_all(common)
    if alpha is not None and alpha_min is not None:
        alpha_resized = resize_nearest(np.asarray(alpha, dtype=np.float32), prediction.shape)
        normal_valid &= _interior_all(np.isfinite(alpha_resized) & (alpha_resized >= float(alpha_min)))
    normal_count = int(normal_valid.sum())
    result["normal_valid_pixels"] = normal_count
    gt_normal_count = int((gt_normal_valid & _interior_all(gt_valid)).sum())
    result["normal_coverage"] = float(normal_count / gt_normal_count) if gt_normal_count else math.nan
    if normal_count:
        dots = np.sum(pred_normals[normal_valid] * gt_normals[normal_valid], axis=-1)
        dots = np.clip(dots, -1.0, 1.0)
        result["normal_mae_deg"] = float(np.degrees(np.mean(np.arccos(dots))))
    return result


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_prediction_root(path: Path) -> tuple[Path, dict[str, Any] | None]:
    path = Path(path).expanduser().resolve()
    candidates = [path]
    if (path / "geometry_raw").is_dir():
        candidates.insert(0, path / "geometry_raw")
    raw = next((candidate for candidate in candidates if (candidate / "depth_expected").is_dir()), None)
    if raw is None:
        raise FileNotFoundError(
            f"Cannot find geometry_raw/depth_expected below prediction root: {path}"
        )
    manifest = _load_json(raw / "geometry_manifest.json") or _load_json(raw.parent / "geometry_manifest.json")
    return raw, manifest


def _list_prediction_frames(path: Path) -> list[PredictionFrame]:
    raw, manifest = _resolve_prediction_root(path)
    frames: list[PredictionFrame] = []
    manifest_frames = manifest.get("frames", []) if manifest else []
    if isinstance(manifest_frames, list) and manifest_frames:
        for position, item in enumerate(manifest_frames):
            if not isinstance(item, Mapping):
                continue
            depth_name = item.get("depth_expected") or item.get("depth_path")
            if not depth_name:
                continue
            depth_path = Path(str(depth_name))
            if not depth_path.is_absolute():
                depth_path = raw / depth_path
            frame_name = str(item.get("frame_name") or item.get("name") or f"{position:05d}")
            try:
                frame_index = int(item.get("frame_index", item.get("index", position)))
            except (TypeError, ValueError):
                frame_index = position
            alpha_name = item.get("alpha") or item.get("alpha_path")
            alpha_path = None
            if alpha_name:
                alpha_path = Path(str(alpha_name))
                if not alpha_path.is_absolute():
                    alpha_path = raw / alpha_path
            default_alpha = raw / "alpha" / f"{Path(frame_name).stem}.npy"
            if alpha_path is None and default_alpha.is_file():
                alpha_path = default_alpha
            if depth_path.is_file():
                frames.append(PredictionFrame(frame_index, frame_name, depth_path, alpha_path if alpha_path and alpha_path.is_file() else None))
    if not frames:
        depth_files = sorted((raw / "depth_expected").glob("*.npy"), key=_numeric_key)
        for position, depth_path in enumerate(depth_files):
            frame_name = depth_path.stem
            alpha_path = raw / "alpha" / f"{frame_name}.npy"
            try:
                frame_index = int(frame_name)
            except ValueError:
                frame_index = position
            frames.append(PredictionFrame(frame_index, frame_name, depth_path, alpha_path if alpha_path.is_file() else None))
    if not frames:
        raise FileNotFoundError(f"No prediction arrays found in {raw / 'depth_expected'}")
    frames = sorted(frames, key=lambda frame: (frame.frame_index, _numeric_key(frame.frame_name)))
    frame_indices = [frame.frame_index for frame in frames]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"Prediction frame indices must be unique, got {frame_indices!r}")
    return frames


def _find_frame_map(sequence_path: Path) -> tuple[Path, dict[str, Any]] | None:
    sequence_path = Path(sequence_path).expanduser().resolve()
    candidates = [sequence_path / "frame_map.json", sequence_path.parent / "frame_map.json", sequence_path.parent.parent / "frame_map.json"]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        value = _load_json(candidate)
        if value and isinstance(value.get("frames"), list):
            return candidate, value
    return None


def _resolve_map_path(map_path: Path, value: str, source_dataset: str | None = None) -> Path:
    normalized_value = value.replace("\\", "/")
    candidate = Path(normalized_value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    local = (map_path.parent / candidate).resolve()
    if local.is_file():
        return local
    if source_dataset:
        source = Path(source_dataset)
        if source.is_dir():
            source_candidate = (source / candidate).resolve()
            if source_candidate.is_file():
                return source_candidate
    return local


def _gt_paths_from_map(map_path: Path, payload: Mapping[str, Any]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    source_dataset = payload.get("source_dataset")
    for position, item in enumerate(payload.get("frames", [])):
        if not isinstance(item, Mapping):
            continue
        try:
            index = int(item.get("frame_index", item.get("index", position)))
        except (TypeError, ValueError):
            index = position
        value = item.get("depth_output_path") or item.get("depth_source_path")
        if value:
            candidate = _resolve_map_path(map_path, str(value), str(source_dataset) if source_dataset else None)
            if candidate.is_file():
                result[index] = candidate
    return result


def _paths_from_tum_index(root: Path) -> list[Path]:
    index_path = root / "depth.txt"
    if not index_path.is_file():
        return []
    paths: list[Path] = []
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        # Normalize the TUM slash-separated relative path for Windows and
        # POSIX callers alike.
        relative = Path(*fields[1].replace("\\", "/").split("/"))
        candidate = root / relative
        if candidate.is_file():
            paths.append(candidate)
    return paths


def _candidate_gt_paths(sequence_path: Path, gt_root: Path | None) -> list[Path]:
    if gt_root is not None:
        root = Path(gt_root).expanduser().resolve()
        if root.is_file():
            return [root]
        tum_paths = _paths_from_tum_index(root)
        if tum_paths:
            return tum_paths
        return sorted(
            [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".npy", ".npz", ".tif", ".tiff", ".pgm"}],
            key=_numeric_key,
        )

    sequence_path = Path(sequence_path).expanduser().resolve()
    map_result = _find_frame_map(sequence_path)
    if map_result:
        map_path, payload = map_result
        mapped = _gt_paths_from_map(map_path, payload)
        if mapped:
            return [mapped[index] for index in sorted(mapped)]
    outer = sequence_path
    if outer.is_dir() and outer.name.lower() not in {"", "."}:
        scene = outer.name
        if (outer / f"{scene}_depth_gt").is_dir():
            return sorted((outer / f"{scene}_depth_gt").glob("*"), key=_numeric_key)
        if (outer.parent / f"{scene}_depth_gt").is_dir():
            return sorted((outer.parent / f"{scene}_depth_gt").glob("*"), key=_numeric_key)
    tum_paths = _paths_from_tum_index(sequence_path)
    if tum_paths:
        return tum_paths
    sibling = sequence_path.parent / f"{sequence_path.name}_depth_gt"
    if sibling.is_dir():
        return sorted(sibling.glob("*"), key=_numeric_key)
    return []


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and math.isnan(value):
                    output[field] = "nan"
                else:
                    output[field] = value
            writer.writerow(output)


FRAME_FIELDS = (
    "sequence",
    "frame_index",
    "frame_name",
    "prediction_path",
    "ground_truth_path",
    "alpha_path",
    "status",
    "scale",
    "gt_valid_pixels",
    "common_valid_pixels",
    "coverage",
    "abs_rel",
    "rmse",
    "delta1",
    "normal_mae_deg",
    "normal_valid_pixels",
    "normal_coverage",
)
SEQUENCE_FIELDS = (
    "sequence",
    "frame_count",
    "valid_frame_count",
    "scale",
    "scale_frame_index",
    "coverage",
    "abs_rel",
    "rmse",
    "delta1",
    "normal_mae_deg",
    "normal_coverage",
    "gt_valid_pixels",
    "common_valid_pixels",
)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get("status") == "ok" and math.isfinite(float(row.get(key, math.nan)))]
    return float(np.mean(values)) if values else math.nan


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf values to JSON null recursively."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def evaluate_sequence(
    prediction_root: Path | str,
    sequence_path: Path | str,
    output_dir: Path | str | None = None,
    gt_root: Path | str | None = None,
    *,
    intrinsics: Intrinsics | Mapping[str, float] | Sequence[float] = BONN_INTRINSICS,
    intrinsics_size: tuple[int, int] = BONN_IMAGE_SIZE,
    alpha_min: float | None = 0.01,
    min_depth: float = 1e-6,
    gt_png_scale: float = 5000.0,
    min_scale_pixels: int = 100,
) -> dict[str, Any]:
    """Evaluate one trained sequence and write machine-readable artifacts."""

    prediction_root = Path(prediction_root).expanduser().resolve()
    sequence_path = Path(sequence_path).expanduser().resolve()
    frames = _list_prediction_frames(prediction_root)
    gt_paths = _candidate_gt_paths(sequence_path, Path(gt_root).expanduser().resolve() if gt_root else None)
    # An explicitly supplied GT root is authoritative; only auto-discover a
    # frame map when pairing against the sequence's default Bonn layout.
    frame_map_result = _find_frame_map(sequence_path) if gt_root is None else None
    mapped_gt_paths = (
        _gt_paths_from_map(frame_map_result[0], frame_map_result[1])
        if frame_map_result
        else {}
    )
    if not gt_paths:
        raise FileNotFoundError(
            f"No Bonn/TUM depth ground truth found for {sequence_path}; pass --gt-root explicitly"
        )

    sequence_name = sequence_path.name
    if sequence_name.endswith("_5fps"):
        sequence_name = sequence_name[:-5]
    rows: list[dict[str, Any]] = []
    loaded: list[tuple[PredictionFrame, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]] = []
    for position, frame in enumerate(frames):
        # A generated frame_map is authoritative.  Position-only pairing is
        # retained only for raw TUM directories that have no frame map.
        if frame_map_result:
            gt_path = mapped_gt_paths.get(frame.frame_index)
        else:
            gt_path = gt_paths[position] if position < len(gt_paths) else None
        row: dict[str, Any] = {
            "sequence": sequence_name,
            "frame_index": frame.frame_index,
            "frame_name": frame.frame_name,
            "prediction_path": str(frame.depth_path),
            "ground_truth_path": str(gt_path) if gt_path else "",
            "alpha_path": str(frame.alpha_path) if frame.alpha_path else "",
        }
        if gt_path is None:
            row.update({"status": "missing_groundtruth", "scale": math.nan})
            rows.append(row)
            continue
        prediction = load_depth(frame.depth_path, npy_scale=1.0)
        ground_truth = load_depth(gt_path, png_scale=gt_png_scale, npy_scale=1.0)
        alpha = load_alpha(frame.alpha_path) if frame.alpha_path else None
        prepared_prediction, prepared_gt, gt_valid, common = _prepare_pair(
            prediction, ground_truth, alpha, alpha_min, min_depth
        )
        loaded.append((frame, prepared_prediction, prepared_gt, alpha, gt_valid, common))
        row["_gt_path"] = gt_path
        rows.append(row)

    scale = None
    scale_frame_index = None
    scale_error = None
    for frame, prediction, ground_truth, _alpha, _gt_valid, common in loaded:
        try:
            scale = estimate_scale(prediction, ground_truth, common, min_scale_pixels)
            scale_frame_index = frame.frame_index
            break
        except ValueError as exc:
            scale_error = str(exc)
    if scale is None:
        raise ValueError(f"Unable to estimate a sequence scale from any frame: {scale_error or 'no frames'}")

    loaded_by_name = {(item[0].frame_index, item[0].frame_name): item for item in loaded}
    finalized_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["frame_index"]), str(row["frame_name"]))
        item = loaded_by_name.get(key)
        if item is None:
            row.setdefault("status", "missing_groundtruth")
            finalized_rows.append(row)
            continue
        frame, prediction, ground_truth, alpha, _gt_valid, _common = item
        metrics = compute_frame_metrics(
            prediction,
            ground_truth,
            alpha,
            scale=scale,
            intrinsics=Intrinsics.from_value(intrinsics),
            intrinsics_size=intrinsics_size,
            alpha_min=alpha_min,
            min_depth=min_depth,
            min_scale_pixels=min_scale_pixels,
        )
        row.update(metrics)
        finalized_rows.append(row)

    valid_rows = [row for row in finalized_rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "sequence": sequence_name,
        "frame_count": len(finalized_rows),
        "valid_frame_count": len(valid_rows),
        "scale": float(scale),
        "scale_frame_index": scale_frame_index,
        "coverage": _mean(finalized_rows, "coverage"),
        "abs_rel": _mean(finalized_rows, "abs_rel"),
        "rmse": _mean(finalized_rows, "rmse"),
        "delta1": _mean(finalized_rows, "delta1"),
        "normal_mae_deg": _mean(finalized_rows, "normal_mae_deg"),
        "normal_coverage": _mean(finalized_rows, "normal_coverage"),
        "gt_valid_pixels": int(sum(int(row.get("gt_valid_pixels", 0)) for row in valid_rows)),
        "common_valid_pixels": int(sum(int(row.get("common_valid_pixels", 0)) for row in valid_rows)),
    }

    raw_root, raw_manifest = _resolve_prediction_root(prediction_root)
    destination = Path(output_dir).expanduser().resolve() if output_dir else raw_root.parent / "geometry_eval"
    destination.mkdir(parents=True, exist_ok=True)
    # Internal fields are useful while matching but should not leak into CSV.
    csv_rows = [{field: row.get(field, "") for field in FRAME_FIELDS} for row in finalized_rows]
    _write_csv(destination / "frame_metrics.csv", csv_rows, FRAME_FIELDS)
    _write_csv(destination / "sequence_metrics.csv", [summary], SEQUENCE_FIELDS)
    single_sequence_macro = dict(summary)
    single_sequence_macro["sequence"] = "macro_average"
    single_sequence_macro["scale"] = math.nan
    single_sequence_macro["scale_frame_index"] = ""
    _write_csv(destination / "macro_metrics.csv", [single_sequence_macro], SEQUENCE_FIELDS)

    protocol = {
        "format": "gflow-geometry-eval-v1",
        "sequence": sequence_name,
        "prediction_root": str(raw_root),
        "sequence_path": str(sequence_path),
        "ground_truth_root": str(gt_root) if gt_root else None,
        "ground_truth_encoding": "uint16 PNG / gt_png_scale" if any(Path(row.get("ground_truth_path", "")).suffix.lower() == ".png" for row in finalized_rows) else "metric array",
        "gt_png_scale": gt_png_scale,
        "alpha_min": alpha_min,
        "min_depth": min_depth,
        "scale_mode": "single multiplicative median scale from first frame with enough valid pixels",
        "scale_frame_index": scale_frame_index,
        "scale": scale,
        "normal_reference": "Bonn RGB calibration; normals are derived from registered sensor depth",
        "intrinsics_source_size": {"width": intrinsics_size[0], "height": intrinsics_size[1]},
        "intrinsics": {
            "fx": Intrinsics.from_value(intrinsics).fx,
            "fy": Intrinsics.from_value(intrinsics).fy,
            "cx": Intrinsics.from_value(intrinsics).cx,
            "cy": Intrinsics.from_value(intrinsics).cy,
        },
        "aggregation": "arithmetic mean over valid frame metrics",
        "frame_map_used": bool(frame_map_result),
        "prediction_manifest_format": raw_manifest.get("format") if raw_manifest else None,
    }
    (destination / "protocol.json").write_text(json.dumps(_json_safe(protocol), indent=2) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "output_dir": str(destination), "frame_metrics": str(destination / "frame_metrics.csv"), "sequence_metrics": str(destination / "sequence_metrics.csv"), "protocol": str(destination / "protocol.json")}


def write_macro_metrics(summaries: Sequence[Mapping[str, Any]], output_path: Path | str) -> Path:
    """Write a macro-average CSV from one or more sequence summaries."""

    summaries = list(summaries)
    if not summaries:
        raise ValueError("At least one sequence summary is required")
    row: dict[str, Any] = {"sequence": "macro_average", "frame_count": len(summaries), "valid_frame_count": int(sum(int(item.get("valid_frame_count", 0)) for item in summaries))}
    for field in SEQUENCE_FIELDS:
        if field in {"sequence", "frame_count", "valid_frame_count", "scale_frame_index", "gt_valid_pixels", "common_valid_pixels"}:
            continue
        values = [float(item[field]) for item in summaries if math.isfinite(float(item.get(field, math.nan)))]
        row[field] = float(np.mean(values)) if values else math.nan
    row["scale"] = math.nan
    row["scale_frame_index"] = ""
    row["gt_valid_pixels"] = int(sum(int(item.get("gt_valid_pixels", 0)) for item in summaries))
    row["common_valid_pixels"] = int(sum(int(item.get("common_valid_pixels", 0)) for item in summaries))
    output_path = Path(output_path).expanduser().resolve()
    _write_csv(output_path, [row], SEQUENCE_FIELDS)
    return output_path
