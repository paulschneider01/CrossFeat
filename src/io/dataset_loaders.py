"""Per-dataset manifest builders.

Each builder is registered via @register_dataset and produces a DatasetManifest
with SampleRecord entries. All datasets (including ReMIND) go through this path.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from src.io.dataset_registry import (
    DatasetManifest,
    SampleRecord,
    register_dataset,
)
from src.io.nifti_orient import get_axial_dim
from src.utils.roi_preprocessing import compute_safe_crop_bounds

logger = logging.getLogger(__name__)

_PAIRED_FOLDER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


def _compute_image_diagonal(records: list[SampleRecord]) -> float | None:
    """Compute image diagonal from the first record's first modality path.

    Returns:
        Image diagonal in pixels, or None if computation fails.
    """
    if not records:
        return None
    first_path = next(iter(records[0].modality_paths.values()))
    try:
        if first_path.suffix == ".npy":
            arr = np.load(str(first_path))
            if arr.ndim == 2:
                h, w = arr.shape
            elif arr.ndim == 3:
                # Detect channel-first (C, H, W) vs channel-last (H, W, C)
                if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4 and arr.shape[2] > 4:
                    h, w = arr.shape[1], arr.shape[2]
                elif arr.shape[2] in (1, 3, 4):
                    h, w = arr.shape[0], arr.shape[1]
                else:
                    h, w = arr.shape[:2]
            else:
                h, w = arr.shape[:2]
        else:
            img = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            h, w = img.shape[:2]
        return math.sqrt(h * h + w * w)
    except (OSError, ValueError, IndexError) as exc:
        logger.debug("Could not compute image diagonal: %s", exc)
        return None


def _resolve_nifti_path(case_dir: Path, case_id: str, vol_name: str) -> Path:
    """Resolve NIfTI path, trying .nii then .nii.gz.

    Raises:
        FileNotFoundError: If neither .nii nor .nii.gz exists.
    """
    nii_path = case_dir / "img" / f"{case_id}-{vol_name}.nii"
    nii_gz_path = case_dir / "img" / f"{case_id}-{vol_name}.nii.gz"
    if nii_path.exists():
        return nii_path
    if nii_gz_path.exists():
        return nii_gz_path
    raise FileNotFoundError(
        f"Neither {nii_path} nor {nii_gz_path} exists "
        f"for case={case_id}, modality={vol_name}"
    )


def _resolve_brats_nifti_path(case_dir: Path, case_id: str, modality: str) -> Path:
    """Resolve BRATS path ``{case_id}_{modality}.nii[.gz]`` with alias fallback."""
    from src.io.modality_aliases import resolve_reverse_alias

    for mod_name in (modality, resolve_reverse_alias(modality)):
        nii_path = case_dir / f"{case_id}_{mod_name}.nii"
        nii_gz_path = case_dir / f"{case_id}_{mod_name}.nii.gz"
        if nii_path.exists():
            return nii_path
        if nii_gz_path.exists():
            return nii_gz_path
    raise FileNotFoundError(
        f"No NIfTI found for case={case_id}, modality={modality} "
        f"(also tried '{resolve_reverse_alias(modality)}')"
    )


def _list_paired_folder_files(directory: Path) -> dict[Path, Path]:
    """Return supported files keyed by their path relative to ``directory``."""
    return {
        path.relative_to(directory): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in _PAIRED_FOLDER_EXTENSIONS
    }


def _path_preview(paths: set[Path], limit: int = 5) -> str:
    preview = ", ".join(path.as_posix() for path in sorted(paths)[:limit])
    if len(paths) > limit:
        preview += f", ... ({len(paths) - limit} more)"
    return preview


@register_dataset("paired_folders")
def paired_folders_manifest(
    data_root: Path,
    modalities: list[str],
) -> DatasetManifest:
    """Build a manifest from aligned image folders.

    The expected layout is ``root/{train,val,test}/{modality}/RELATIVE_PATH``.
    Each split must contain exactly the same relative paths for both modalities.
    Images are also required to have equal dimensions when they are loaded.
    """
    if len(modalities) != 2:
        raise ValueError(
            "paired_folders requires exactly two modalities; "
            f"received {modalities!r}"
        )
    if modalities[0] == modalities[1]:
        raise ValueError("paired_folders modality names must be different")

    records: list[SampleRecord] = []
    for split in ("train", "val", "test"):
        files_by_modality: dict[str, dict[Path, Path]] = {}
        for modality in modalities:
            directory = data_root / split / modality
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Missing paired_folders directory for split={split!r}, "
                    f"modality={modality!r}: {directory}"
                )
            files = _list_paired_folder_files(directory)
            if not files:
                extensions = ", ".join(sorted(_PAIRED_FOLDER_EXTENSIONS))
                raise ValueError(
                    f"No supported files under {directory}. Supported: {extensions}"
                )
            files_by_modality[modality] = files

        reference_paths = set(files_by_modality[modalities[0]])
        other_paths = set(files_by_modality[modalities[1]])
        missing_other = reference_paths - other_paths
        missing_reference = other_paths - reference_paths
        if missing_other or missing_reference:
            details: list[str] = []
            if missing_other:
                details.append(
                    f"missing from {modalities[1]!r}: {_path_preview(missing_other)}"
                )
            if missing_reference:
                details.append(
                    f"missing from {modalities[0]!r}: {_path_preview(missing_reference)}"
                )
            raise ValueError(
                f"Unpaired files in split={split!r}; " + "; ".join(details)
            )

        for relative_path in sorted(reference_paths):
            records.append(
                SampleRecord(
                    sample_id=f"{split}/{relative_path.as_posix()}",
                    modality_paths={
                        modality: files_by_modality[modality][relative_path]
                        for modality in modalities
                    },
                    split=split,
                    group_key=relative_path.as_posix(),
                    metadata={"require_same_shape": True, "max_dim": None},
                )
            )

    return DatasetManifest(
        name="paired_folders",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        image_diagonal=_compute_image_diagonal(records),
        records=records,
    )


@register_dataset("remind")
def remind_manifest(
    data_root: Path,
    modalities: list[str],
    use_crop: bool | None = None,
    splits_file: str | Path | None = None,
) -> DatasetManifest:
    """Build manifest for ReMIND dataset (slice-level records with z-range intersection).

    Args:
        data_root: Path to ReMIND data directory.
        modalities: List of modality names (e.g., ["t2", "cet1"]).
        use_crop: If False, use _full suffix for non-US modalities. If None, auto-detect.
        splits_file: Path to data_splits.json (default: config/data_splits.json).
    """
    # Load splits
    if splits_file is None:
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        splits_file = project_root / "config" / "data_splits.json"
    splits_file = Path(splits_file)
    with open(splits_file) as f:
        splits = json.load(f)["splits"]

    # Build case_id -> split mapping
    split_for_case: dict[str, str] = {}
    for split_name, case_ids in splits.items():
        for cid in case_ids:
            split_for_case[cid] = split_name

    records: list[SampleRecord] = []

    for case_id, split in sorted(split_for_case.items()):
        case_dir = data_root / case_id
        if not case_dir.exists():
            logger.info("Skipping %s: directory not found", case_id)
            continue

        # Resolve volume paths (with _full suffix and .nii/.nii.gz fallback)
        mod_paths: dict[str, Path] = {}
        skip = False
        for mod in modalities:
            # us never has _full suffix; for other modalities, use _full when use_crop is False
            if use_crop is False and mod != "us":
                vol_name = f"{mod}_full"
            else:
                vol_name = mod
            try:
                mod_paths[mod] = _resolve_nifti_path(case_dir, case_id, vol_name)
            except FileNotFoundError:
                logger.info("Skipping %s: missing modality %s", case_id, mod)
                skip = True
                break
        if skip:
            continue

        # Determine valid z-range from INTERSECTION of all modalities
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError("nibabel is required for ReMIND manifest") from e

        z_ranges = []
        for mod in modalities:
            _img = nib.load(str(mod_paths[mod]))
            shape_z = get_axial_dim(
                np.asarray(_img.affine, dtype=np.float64),
                _img.header.get_data_shape(),
            )
            z_min_arr, z_max_arr = compute_safe_crop_bounds((int(shape_z),), base_margin=15)
            z_ranges.append((int(z_min_arr[0]), int(z_max_arr[0])))
        z_start = max(r[0] for r in z_ranges)
        z_end = min(r[1] for r in z_ranges)

        if z_start >= z_end:
            logger.warning("Skipping %s: empty z-intersection across modalities", case_id)
            continue

        # Create one SampleRecord per valid z-slice
        for z in range(z_start, z_end):
            records.append(
                SampleRecord(
                    sample_id=f"{case_id}:z{z:03d}",
                    modality_paths=dict(mod_paths),
                    split=split,
                    group_key=case_id,
                    slice_idx=z,
                )
            )

    return DatasetManifest(
        name="remind",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        records=records,
    )


# Maps DELIVER directory names to the filename suffix used inside each.
# Note: "hha" files are named with "_depth_" suffix (HHA is derived from depth
# maps in the DELIVER dataset, so they share the depth filename convention).
_DELIVER_FILENAME_SUFFIX: dict[str, str] = {
    "img": "rgb",
    "depth": "depth",
    "event": "event",
    "hha": "depth",
    "lidar": "lidar",
    "semantic": "semantic",
}


@register_dataset("deliver")
def deliver_manifest(
    data_root: Path,
    modalities: list[str],
    condition: str = "sun",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetManifest:
    """Build manifest for DELIVER dataset.

    Actual structure: ``data_root/{modality}/{condition}/{split}/{scene}/{frame}_{suffix}_front.png``

    Frames share a numeric prefix across modalities within the same
    (condition, split, scene). The ``condition`` parameter selects the weather
    condition (default ``"sun"`` for clear weather).

    Carves val from train by scene to prevent leakage.

    Args:
        data_root: Path to DELIVER extracted directory.
        modalities: List of modality names (e.g., ["img", "depth"]).
        condition: Weather condition to use (cloud, fog, night, rain, sun).
        val_fraction: Fraction of train scenes to hold out for val.
        seed: Random seed for val carving.
    """
    records: list[SampleRecord] = []

    # Discover frames using the first modality, then verify across all
    first_mod = modalities[0]

    for orig_split in ["train", "test"]:
        first_mod_dir = data_root / first_mod / condition / orig_split
        if not first_mod_dir.exists():
            logger.debug("Missing DELIVER dir: %s", first_mod_dir)
            continue

        for scene_dir in sorted(first_mod_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name

            # Skip scenes with known resolution mismatches (e.g.
            # *_eventlowres scenes have 512x512 event images but
            # 1042x1042 for all other modalities).
            if "lowres" in scene:
                continue

            for frame_file in sorted(scene_dir.glob("*_*_front.png")):
                # Extract numeric frame prefix: "041900_rgb_front.png" -> "041900"
                frame_num = frame_file.stem.split("_")[0]
                sample_id = f"{condition}/{orig_split}/{scene}/{frame_num}"

                mod_paths: dict[str, Path] = {}
                skip = False
                for mod in modalities:
                    suffix = _DELIVER_FILENAME_SUFFIX.get(mod, mod)
                    mod_path = (
                        data_root / mod / condition / orig_split
                        / scene / f"{frame_num}_{suffix}_front.png"
                    )
                    if not mod_path.exists():
                        skip = True
                        break
                    mod_paths[mod] = mod_path

                if skip:
                    continue

                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        modality_paths=mod_paths,
                        split=orig_split,
                        group_key=scene,
                    )
                )

    # Carve val from train by scene
    rng = np.random.RandomState(seed)
    train_scenes = sorted({r.group_key for r in records if r.split == "train"})
    n_val_scenes = max(1, int(len(train_scenes) * val_fraction))
    rng.shuffle(train_scenes)
    val_scenes = set(train_scenes[:n_val_scenes])

    for r in records:
        if r.split == "train" and r.group_key in val_scenes:
            r.split = "val"

    image_diagonal = _compute_image_diagonal(records)

    return DatasetManifest(
        name="deliver",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        image_diagonal=image_diagonal,
        records=records,
    )


@register_dataset("resect")
def resect_manifest(
    data_root: Path,
    modalities: list[str],
    use_crop: bool | None = None,
) -> DatasetManifest:
    """Build manifest for RESECT dataset (test-only, for cross-dataset eval).

    RESECT layout (matches ReMIND after preprocessing)::

        data_root/
        ├── Case001/
        │   └── img/
        │       ├── Case001-cet1.nii.gz       (T1 masked to US FOV)
        │       ├── Case001-cet1_full.nii.gz   (T1 full)
        │       ├── Case001-us.nii.gz          (US)
        │       └── Case001-us_full.nii.gz     (US, identical)
        ...

    All cases are assigned to the "test" split since RESECT is used only for
    cross-dataset evaluation (train on ReMIND, test on RESECT).

    Args:
        data_root: Path to RESECT_ALIGNED data directory.
        modalities: List of modality names (e.g., ["cet1", "us"]).
        use_crop: If False, use _full suffix for non-US modalities. If None,
            auto-detect (defaults to True for US-containing pairs).
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required for RESECT manifest") from e

    records: list[SampleRecord] = []

    # Discover Case### directories
    case_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("Case")
    )

    if not case_dirs:
        logger.warning("No Case* directories found in %s", data_root)
        return DatasetManifest(
            name="resect",
            root=data_root,
            modalities=list(modalities),
            is_2d=True,
            records=[],
        )

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Resolve volume paths (same layout as ReMIND: Case###/img/Case###-{mod}.nii.gz)
        mod_paths: dict[str, Path] = {}
        skip = False
        for mod in modalities:
            if use_crop is False and mod != "us":
                vol_name = f"{mod}_full"
            else:
                vol_name = mod
            try:
                mod_paths[mod] = _resolve_nifti_path(case_dir, case_id, vol_name)
            except FileNotFoundError:
                logger.info("Skipping %s: missing modality %s", case_id, mod)
                skip = True
                break
        if skip:
            continue

        # Determine valid z-range from INTERSECTION of all modalities
        z_ranges = []
        for mod in modalities:
            _img = nib.load(str(mod_paths[mod]))
            shape_z = get_axial_dim(
                np.asarray(_img.affine, dtype=np.float64),
                _img.header.get_data_shape(),
            )
            z_min_arr, z_max_arr = compute_safe_crop_bounds(
                (int(shape_z),), base_margin=15,
            )
            z_ranges.append((int(z_min_arr[0]), int(z_max_arr[0])))
        z_start = max(r[0] for r in z_ranges)
        z_end = min(r[1] for r in z_ranges)

        if z_start >= z_end:
            logger.warning(
                "Skipping %s: empty z-intersection across modalities", case_id,
            )
            continue

        # Create one SampleRecord per valid z-slice (all as "test")
        for z in range(z_start, z_end):
            records.append(
                SampleRecord(
                    sample_id=f"{case_id}:z{z:03d}",
                    modality_paths=dict(mod_paths),
                    split="test",
                    group_key=case_id,
                    slice_idx=z,
                )
            )

    logger.info(
        "RESECT manifest: %d records from %d cases (all test)",
        len(records),
        len({r.group_key for r in records}),
    )

    return DatasetManifest(
        name="resect",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        records=records,
    )


@register_dataset("brats")
def brats_manifest(
    data_root: Path,
    modalities: list[str],
    splits_file: str | Path | None = None,
) -> DatasetManifest:
    """Build manifest for BraTS20 directories, producing slice-level records."""
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required for BRATS manifest") from e

    split_for_case: dict[str, str] = {}
    if splits_file is not None:
        with open(Path(splits_file), encoding="utf-8") as f:
            splits = json.load(f)["splits"]
        for split_name, case_ids in splits.items():
            for case_id in case_ids:
                split_for_case[str(case_id)] = str(split_name)
    else:
        for split_dir, split_name in (("train", "train"), ("val", "val"), ("test", "test")):
            root = data_root / split_dir
            if not root.exists():
                continue
            for case_dir in root.iterdir():
                if case_dir.is_dir():
                    split_for_case[case_dir.name] = split_name

    records: list[SampleRecord] = []
    for case_id, split in sorted(split_for_case.items()):
        case_dir = None
        for split_dir in ("train", "val", "test"):
            candidate = data_root / split_dir / case_id
            if candidate.exists():
                case_dir = candidate
                break
        if case_dir is None:
            continue

        mod_paths: dict[str, Path] = {}
        missing = False
        for mod in modalities:
            try:
                mod_paths[mod] = _resolve_brats_nifti_path(case_dir, case_id, mod)
            except FileNotFoundError:
                missing = True
                break
        if missing:
            continue

        z_ranges: list[tuple[int, int]] = []
        for mod in modalities:
            img = nib.load(str(mod_paths[mod]))
            shape_z = get_axial_dim(
                np.asarray(img.affine, dtype=np.float64),
                img.header.get_data_shape(),
            )
            z_min_arr, z_max_arr = compute_safe_crop_bounds((int(shape_z),), base_margin=15)
            z_ranges.append((int(z_min_arr[0]), int(z_max_arr[0])))
        z_start = max(r[0] for r in z_ranges)
        z_end = min(r[1] for r in z_ranges)
        if z_start >= z_end:
            continue

        for z in range(z_start, z_end):
            records.append(
                SampleRecord(
                    sample_id=f"{case_id}:z{z:03d}",
                    modality_paths=dict(mod_paths),
                    split=split,
                    group_key=case_id,
                    slice_idx=z,
                )
            )

    return DatasetManifest(
        name="brats",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        records=records,
    )


# Maps QXS-SAROPT modality names to their directory names on disk.
_QXS_SAROPT_DIR: dict[str, str] = {
    "sar": "sar_256_oc_0.2",
    "opt": "opt_256_oc_0.2",
    "optical": "opt_256_oc_0.2",
    "rgb": "opt_256_oc_0.2",
    "img": "opt_256_oc_0.2",
}


@register_dataset("qxs_saropt")
def qxs_saropt_manifest(
    data_root: Path,
    modalities: list[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetManifest:
    """Build manifest for QXS-SAROPT dataset (SAR-optical remote sensing patches).

    Expects structure::

        data_root/
        ├── sar_256_oc_0.2/   # SAR images (256x256 grayscale PNG)
        │   ├── 1.png
        │   └── ...20000.png
        └── opt_256_oc_0.2/   # Optical/RGB images (256x256 3-channel PNG)
            ├── 1.png
            └── ...20000.png

    Pairs matched by filename number. No predefined split files; uses seeded
    random 70/15/15 split by image ID.

    Args:
        data_root: Path to QXSLAB_SAROPT directory.
        modalities: List of modality names (e.g., ["sar", "opt"]).
        val_fraction: Fraction for validation (default 0.15).
        test_fraction: Fraction for test (default 0.15).
        seed: Random seed for deterministic split.
    """
    # Discover image pairs from first modality directory
    first_mod = modalities[0]
    first_dir_name = _QXS_SAROPT_DIR.get(first_mod, first_mod)
    first_dir = data_root / first_dir_name

    if not first_dir.exists():
        logger.warning("QXS-SAROPT: missing directory %s", first_dir)
        return DatasetManifest(
            name="qxs_saropt",
            root=data_root,
            modalities=list(modalities),
            is_2d=True,
            records=[],
        )

    # Collect image IDs (numeric stems)
    image_ids = sorted(
        int(f.stem) for f in first_dir.glob("*.png")
    )

    if not image_ids:
        logger.warning("QXS-SAROPT: no PNG files found in %s", first_dir)
        return DatasetManifest(
            name="qxs_saropt",
            root=data_root,
            modalities=list(modalities),
            is_2d=True,
            records=[],
        )

    # Assign splits by shuffled image ID
    rng = np.random.RandomState(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(n_total * val_fraction))
    n_test = max(1, int(n_total * test_fraction))
    val_ids = set(shuffled[:n_val])
    test_ids = set(shuffled[n_val : n_val + n_test])

    records: list[SampleRecord] = []
    skipped = 0

    for img_id in image_ids:
        fname = f"{img_id}.png"

        mod_paths: dict[str, Path] = {}
        skip = False
        for mod in modalities:
            dir_name = _QXS_SAROPT_DIR.get(mod, mod)
            mod_path = data_root / dir_name / fname
            if not mod_path.exists():
                skip = True
                break
            mod_paths[mod] = mod_path

        if skip:
            skipped += 1
            continue

        if img_id in val_ids:
            split = "val"
        elif img_id in test_ids:
            split = "test"
        else:
            split = "train"

        sample_id = str(img_id)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                modality_paths=mod_paths,
                split=split,
                group_key=sample_id,
            )
        )

    if skipped:
        logger.warning(
            "QXS-SAROPT: skipped %d/%d images (missing modality files)",
            skipped, len(image_ids),
        )

    image_diagonal = _compute_image_diagonal(records)

    logger.info(
        "QXS-SAROPT manifest: %d records (train=%d, val=%d, test=%d)",
        len(records),
        sum(1 for r in records if r.split == "train"),
        sum(1 for r in records if r.split == "val"),
        sum(1 for r in records if r.split == "test"),
    )

    return DatasetManifest(
        name="qxs_saropt",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        image_diagonal=image_diagonal,
        records=records,
    )


# ---------------------------------------------------------------------------
# WHU-OPT-SAR
# ---------------------------------------------------------------------------

# Maps WHU-OPT-SAR modality names to their directory names on disk.
_WHU_OPT_SAR_DIR: dict[str, str] = {
    "optical": "optical",
    "opt": "optical",
    "sar": "sar",
}


@register_dataset("whu_opt_sar")
def whu_opt_sar_manifest(
    data_root: Path,
    modalities: list[str],
    patch_size: int = 1024,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetManifest:
    """Build manifest for WHU-OPT-SAR dataset (optical-SAR remote sensing).

    Expects structure::

        data_root/
        ├── optical/   # 4-channel GeoTIFF (RGB + NIR), 3704×5556
        │   ├── NH49E012020.tif
        │   └── ...
        └── sar/       # 1-channel GeoTIFF (grayscale), 3704×5556
            ├── NH49E012020.tif
            └── ...

    Images are large (3704×5556) so they are cropped into non-overlapping
    patches of ``patch_size × patch_size``.  All patches from the same source
    image stay in the same split to prevent data leakage.

    Args:
        data_root: Path to WHU-OPT-SAR directory.
        modalities: List of modality names (e.g., ["optical", "sar"]).
        patch_size: Patch size for cropping large images (default 1024).
        val_fraction: Fraction for validation (default 0.15).
        test_fraction: Fraction for test (default 0.15).
        seed: Random seed for deterministic split.
    """
    # Discover paired images (intersection of stems across modalities)
    stems_per_mod: list[set[str]] = []
    for mod in modalities:
        dir_name = _WHU_OPT_SAR_DIR.get(mod, mod)
        mod_dir = data_root / dir_name
        if not mod_dir.exists():
            logger.warning("WHU-OPT-SAR: missing directory %s", mod_dir)
            return DatasetManifest(
                name="whu_opt_sar",
                root=data_root,
                modalities=list(modalities),
                is_2d=True,
                records=[],
            )
        stems_per_mod.append({f.stem for f in mod_dir.glob("*.tif")})

    common_stems = sorted(set.intersection(*stems_per_mod))

    if not common_stems:
        logger.warning("WHU-OPT-SAR: no paired images found")
        return DatasetManifest(
            name="whu_opt_sar",
            root=data_root,
            modalities=list(modalities),
            is_2d=True,
            records=[],
        )

    # Read one image to get spatial dimensions
    first_mod = modalities[0]
    first_dir = _WHU_OPT_SAR_DIR.get(first_mod, first_mod)
    sample_path = data_root / first_dir / f"{common_stems[0]}.tif"
    sample_img = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if sample_img is None:
        logger.warning("WHU-OPT-SAR: cannot read %s", sample_path)
        return DatasetManifest(
            name="whu_opt_sar",
            root=data_root,
            modalities=list(modalities),
            is_2d=True,
            records=[],
        )
    img_h, img_w = sample_img.shape[:2]
    n_rows = img_h // patch_size
    n_cols = img_w // patch_size
    logger.info(
        "WHU-OPT-SAR: image size %dx%d, patch %dx%d -> %d×%d = %d patches/image",
        img_w, img_h, patch_size, patch_size, n_cols, n_rows, n_rows * n_cols,
    )

    # Assign splits by source image (not by patch)
    rng = np.random.RandomState(seed)
    shuffled = list(common_stems)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(n_total * val_fraction))
    n_test = max(1, int(n_total * test_fraction))
    val_stems = set(shuffled[:n_val])
    test_stems = set(shuffled[n_val : n_val + n_test])

    records: list[SampleRecord] = []

    for stem in common_stems:
        if stem in val_stems:
            split = "val"
        elif stem in test_stems:
            split = "test"
        else:
            split = "train"

        # Create one record per patch
        for row in range(n_rows):
            for col in range(n_cols):
                y0 = row * patch_size
                x0 = col * patch_size
                patch_id = f"{stem}:r{row}c{col}"

                mod_paths: dict[str, Path] = {}
                for mod in modalities:
                    dir_name = _WHU_OPT_SAR_DIR.get(mod, mod)
                    mod_paths[mod] = data_root / dir_name / f"{stem}.tif"

                records.append(
                    SampleRecord(
                        sample_id=patch_id,
                        modality_paths=mod_paths,
                        split=split,
                        group_key=stem,
                        metadata={
                            "patch_y0": y0,
                            "patch_x0": x0,
                            "patch_size": patch_size,
                        },
                    )
                )

    patch_diag = math.sqrt(2) * patch_size

    logger.info(
        "WHU-OPT-SAR manifest: %d records from %d images "
        "(train=%d, val=%d, test=%d)",
        len(records),
        len(common_stems),
        sum(1 for r in records if r.split == "train"),
        sum(1 for r in records if r.split == "val"),
        sum(1 for r in records if r.split == "test"),
    )

    return DatasetManifest(
        name="whu_opt_sar",
        root=data_root,
        modalities=list(modalities),
        is_2d=True,
        image_diagonal=patch_diag,
        records=records,
    )


# ---------------------------------------------------------------------------
# EventScape
# ---------------------------------------------------------------------------

# Maps EventScape modality names to their subdirectory and file patterns.
# RGB/depth/semantic use: {subdir}/data/{town}_{seq}_{frame:04d}_{suffix}.{ext}
# Events use: events/frames/{frame:04d}.png
_EVENTSCAPE_MODALITY_INFO: dict[str, dict[str, str]] = {
    "rgb": {"subdir": "rgb/data", "pattern": "{prefix}_{frame:04d}_image.png"},
    "depth": {"subdir": "depth/data", "pattern": "{prefix}_{frame:04d}_depth.npy"},
    "events": {"subdir": "events/frames", "pattern": "{frame:04d}.png"},
    "semantic": {
        "subdir": "semantic/data",
        "pattern": "{prefix}_{frame:04d}_gt_labelIds.png",
    },
}


@register_dataset("eventscape")
def eventscape_manifest(
    data_root: Path,
    modalities: list[str],
    stride: int = 5,
    splits_file: str | None = None,
) -> DatasetManifest:
    """Build manifest for EventScape dataset (CARLA synthetic driving data).

    Expects structure::

        data_root/
        ├── Town01/sequence_0/rgb/data/{prefix}_image.png
        │                     /depth/data/{prefix}_depth.npy
        │                     /events/frames/{frame:04d}.png
        │                     /semantic/data/{prefix}_gt_labelIds.png
        ├── Town02/...
        ├── Town03/...  (all train)
        └── Town05/...  (val + test, split by sequence ID)

    Args:
        data_root: Path to EventScape root (contains Town* directories).
        modalities: List of modality names (e.g., ["rgb", "depth"]).
        stride: Keep every Nth frame per sequence (default 5).
        splits_file: Path to JSON with val/test sequence IDs for Town05.
            If None, looks for config/data_splits/eventscape_splits.json.
    """
    # Resolve modality aliases so cross-dataset eval works (e.g. DELIVER
    # uses "event" while EventScape uses "events" on disk).
    _ALIAS_TO_NATIVE: dict[str, str] = {"event": "events"}
    orig_modalities = list(modalities)
    modalities = [_ALIAS_TO_NATIVE.get(m, m) for m in modalities]

    # Load Town05 val/test split
    if splits_file is None:
        candidate = Path("config/data_splits/eventscape_splits.json")
        if not candidate.exists():
            candidate = data_root / "eventscape_splits.json"
        splits_file_path = candidate
    else:
        splits_file_path = Path(splits_file)

    town05_val_ids: set[str] = set()
    town05_test_ids: set[str] = set()
    if splits_file_path.exists():
        with open(splits_file_path) as f:
            splits_data = json.load(f)
        town05_val_ids = set(splits_data.get("val", []))
        town05_test_ids = set(splits_data.get("test", []))
    else:
        logger.warning(
            "EventScape splits file not found: %s. "
            "All Town05 sequences will be assigned to test.",
            splits_file_path,
        )

    records: list[SampleRecord] = []
    has_npy = any(
        _EVENTSCAPE_MODALITY_INFO.get(m, {}).get("pattern", "").endswith(".npy")
        for m in modalities
    )

    for town_dir in sorted(data_root.iterdir()):
        if not town_dir.is_dir() or not town_dir.name.startswith("Town"):
            continue
        town_name = town_dir.name
        is_train_town = town_name in ("Town01", "Town02", "Town03")

        for seq_dir in sorted(town_dir.iterdir()):
            if not seq_dir.is_dir() or not seq_dir.name.startswith("sequence_"):
                continue

            seq_id = seq_dir.name.split("_", 1)[1]  # "sequence_139" -> "139"
            group_key = f"{town_name}/{seq_dir.name}"

            # Determine split
            if is_train_town:
                split = "train"
            elif seq_id in town05_val_ids:
                split = "val"
            elif seq_id in town05_test_ids:
                split = "test"
            else:
                split = "test"

            # Discover frames from the first modality
            first_mod = modalities[0]
            first_info = _EVENTSCAPE_MODALITY_INFO.get(first_mod)
            if first_info is None:
                logger.warning("EventScape: unknown modality %r", first_mod)
                continue

            first_dir = seq_dir / first_info["subdir"]
            if not first_dir.exists():
                continue

            # Extract frame indices from the first modality's files
            frame_indices: list[int] = []
            for f in sorted(first_dir.iterdir()):
                if not f.is_file():
                    continue
                stem = f.stem
                parts = stem.split("_")
                try:
                    if first_mod == "events":
                        idx = int(stem)
                    else:
                        idx = int(parts[2])
                except (ValueError, IndexError):
                    continue
                frame_indices.append(idx)

            frame_indices.sort()
            if stride > 1:
                frame_indices = frame_indices[::stride]

            # Build prefix for non-event modalities
            town_num = town_name.replace("Town", "")
            prefix = f"{town_num}_{seq_id}"

            for fi in frame_indices:
                sample_id = f"{town_name}/{seq_dir.name}/{fi:04d}"

                mod_paths: dict[str, Path] = {}
                skip = False
                for orig_mod, native_mod in zip(orig_modalities, modalities):
                    info = _EVENTSCAPE_MODALITY_INFO.get(native_mod)
                    if info is None:
                        skip = True
                        break
                    pattern = info["pattern"].format(prefix=prefix, frame=fi)
                    mod_path = seq_dir / info["subdir"] / pattern
                    if not mod_path.exists():
                        skip = True
                        break
                    mod_paths[orig_mod] = mod_path

                if skip:
                    continue

                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        modality_paths=mod_paths,
                        split=split,
                        group_key=group_key,
                        normalize_mode="minmax" if has_npy else None,
                    )
                )

    if not records:
        logger.warning("EventScape: no records found under %s", data_root)

    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    n_test = sum(1 for r in records if r.split == "test")
    logger.info(
        "EventScape manifest: %d train, %d val, %d test (stride=%d)",
        n_train, n_val, n_test, stride,
    )

    image_diagonal = _compute_image_diagonal(records)

    return DatasetManifest(
        name="eventscape",
        root=data_root,
        modalities=list(orig_modalities),
        is_2d=True,
        image_diagonal=image_diagonal,
        records=records,
    )
