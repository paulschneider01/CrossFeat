"""SIFT descriptor implementation used by CrossFeat."""

from __future__ import annotations

import functools
import inspect
import threading

import cv2
import numpy as np


def cloneable(cls: type) -> type:
    original_init = cls.__init__

    @functools.wraps(original_init)
    def _tracking_init(self: object, *args: object, **kwargs: object) -> None:
        sig = inspect.signature(original_init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init_kwargs = dict(bound.arguments)
        init_kwargs.pop("self", None)
        object.__setattr__(self, "_init_kwargs", init_kwargs)
        original_init(self, *args, **kwargs)

    cls.__init__ = _tracking_init  # type: ignore[assignment]

    def clone(self: object) -> object:
        return type(self)(**self._init_kwargs)  # type: ignore[attr-defined]

    cls.clone = clone  # type: ignore[attr-defined]
    return cls


def compute_phase_congruency(slice_2d: np.ndarray) -> np.ndarray:
    from phasepack import phasecongmono

    pc, _, _, _ = phasecongmono(slice_2d.astype(np.float64))
    return pc.astype(np.float32)


def compute_gaussian_sobel(
    slice_2d: np.ndarray,
    sigma: float = 7.0,
) -> np.ndarray:
    s = slice_2d.astype(np.float32)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(s, (ksize, ksize), sigma)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(dx**2 + dy**2)


def prepare_slice_uint8(
    slice_2d: np.ndarray,
    normalize_image: bool = True,
    preprocess: str = "none",
) -> np.ndarray:
    s = slice_2d.astype(np.float32)
    if preprocess == "phase_congruency":
        s = compute_phase_congruency(s)
    elif preprocess == "gaussian_sobel":
        s = compute_gaussian_sobel(s)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8) * 255
    s = s.astype(np.uint8)
    if normalize_image:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        s = clahe.apply(s)
    return s

DEFAULT_KEYPOINT_SIZE = 16.0  # Typical SIFT keypoint size for descriptor computation.

# Cache for Gaussian weight kernels: (radius, sigma) -> 2D weight array
_gaussian_kernel_cache: dict[tuple[int, float], np.ndarray] = {}
_gaussian_kernel_lock = threading.Lock()


def _get_gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    """Get cached Gaussian weight kernel centered at (radius, radius).

    Thread-safe: uses double-checked locking pattern. Cached kernels are
    made read-only to prevent accidental mutation across threads.
    """
    key = (radius, sigma)
    # Fast path: check without lock
    existing = _gaussian_kernel_cache.get(key)
    if existing is not None:
        return existing

    # Slow path: acquire lock, double-check, then create
    with _gaussian_kernel_lock:
        existing = _gaussian_kernel_cache.get(key)
        if existing is not None:
            return existing

        size = 2 * radius + 1
        y, x = np.ogrid[:size, :size]
        center = radius
        kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2)).astype(np.float32)
        kernel.setflags(write=False)
        _gaussian_kernel_cache[key] = kernel
        return kernel


def _compute_keypoint_orientations(
    image: np.ndarray,
    keypoints: list,
    cv2_module,
    scale_factor: float = 1.5,
    num_bins: int = 36,
) -> list:
    """
    Compute dominant orientation for keypoints using gradient histograms.

    This replicates SIFT's orientation assignment, enabling rotation invariance
    when using sift.compute() with manually specified keypoint locations.

    Optimized with vectorized histogram binning and cached Gaussian kernels.

    Args:
        image: Grayscale uint8 image
        keypoints: List of cv2.KeyPoint (with angle=-1 or any value)
        cv2_module: OpenCV module reference
        scale_factor: Multiplier for keypoint size to get window radius
        num_bins: Number of orientation histogram bins

    Returns:
        List of cv2.KeyPoint with computed angles (same length as input)
    """
    if len(keypoints) == 0:
        return []

    # Compute gradients once for the whole image
    dx = cv2_module.Sobel(image, cv2_module.CV_64F, 1, 0, ksize=3)
    dy = cv2_module.Sobel(image, cv2_module.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2).astype(np.float32)
    direction = np.arctan2(dy, dx) * (180.0 / np.pi)  # -180 to 180
    direction = ((direction + 360.0) % 360.0).astype(np.float32)  # 0 to 360

    h, w = image.shape
    bin_width = 360.0 / num_bins

    # Precompute smoothing kernel weights [1, 4, 6, 4, 1] / 16
    smooth_weights = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

    oriented_keypoints = []

    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        radius = int(round(kp.size * scale_factor))
        sigma = kp.size * 1.5

        # Extract local region bounds
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        if y_max <= y_min or x_max <= x_min:
            # Invalid region, use upright orientation
            new_kp = cv2_module.KeyPoint(kp.pt[0], kp.pt[1], kp.size, 0.0)
            oriented_keypoints.append(new_kp)
            continue

        local_mag = magnitude[y_min:y_max, x_min:x_max]
        local_dir = direction[y_min:y_max, x_min:x_max]

        # Get Gaussian weight kernel (cached) and extract the relevant portion
        full_kernel = _get_gaussian_kernel(radius, sigma)
        # Compute offsets for kernel slicing (handles boundary cases)
        ky_start = radius - (y - y_min)
        ky_end = ky_start + (y_max - y_min)
        kx_start = radius - (x - x_min)
        kx_end = kx_start + (x_max - x_min)
        weight = full_kernel[ky_start:ky_end, kx_start:kx_end]

        # Weighted magnitudes
        weighted_mag = local_mag * weight

        # Vectorized histogram binning using np.bincount
        bin_indices = ((local_dir / bin_width).astype(np.int32) % num_bins).ravel()
        weights_flat = weighted_mag.ravel()
        hist = np.bincount(bin_indices, weights=weights_flat, minlength=num_bins).astype(np.float32)

        # Vectorized 5-point circular smoothing using np.convolve-like operation
        # Extend histogram circularly for convolution
        hist_extended = np.concatenate([hist[-2:], hist, hist[:2]])
        hist_smooth = np.convolve(hist_extended, smooth_weights, mode='valid')

        # Find dominant peak
        peak_idx = np.argmax(hist_smooth)
        max_val = hist_smooth[peak_idx]

        if max_val < 1e-10:
            # No gradient, use upright
            new_kp = cv2_module.KeyPoint(kp.pt[0], kp.pt[1], kp.size, 0.0)
            oriented_keypoints.append(new_kp)
            continue

        # Parabolic interpolation for sub-bin accuracy
        prev_val = hist_smooth[(peak_idx - 1) % num_bins]
        curr_val = hist_smooth[peak_idx]
        next_val = hist_smooth[(peak_idx + 1) % num_bins]

        denom = prev_val - 2 * curr_val + next_val
        if abs(denom) > 1e-10:
            offset = 0.5 * (prev_val - next_val) / denom
        else:
            offset = 0.0

        angle = ((peak_idx + 0.5 + offset) * bin_width) % 360.0

        new_kp = cv2_module.KeyPoint(kp.pt[0], kp.pt[1], kp.size, angle)
        oriented_keypoints.append(new_kp)

    return oriented_keypoints


def _detect_sift_keypoints_3d(
    extractor,
    volume: np.ndarray,
    roi_mask: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    n_keypoints: int,
) -> np.ndarray:
    """
    Detect SIFT keypoints in a 3D volume by running SIFT detection on 2D slices.

    Notes:
    - Coordinates follow the repo convention (z, y, x).
    - `roi_mask` is a 3D boolean mask indicating valid sampling region (already eroded/cropped upstream).
    - The returned keypoints are deduplicated across slices at integer voxel coordinates; for duplicates,
      the highest SIFT response is kept.
    """
    n_keypoints = int(n_keypoints)
    if n_keypoints <= 0:
        return np.zeros((0, 3), dtype=np.int32)
    if not hasattr(extractor, "sift"):
        raise ValueError("SIFT keypoint detection requires extractor.sift")
    if not hasattr(extractor, "_prepare_slice"):
        raise ValueError("SIFT keypoint detection requires extractor._prepare_slice")

    crop_min = np.asarray(crop_min, dtype=np.int64).reshape(3)
    crop_max = np.asarray(crop_max, dtype=np.int64).reshape(3)
    roi_mask = np.asarray(roi_mask).astype(bool, copy=False)

    z0, z1 = int(crop_min[0]), int(crop_max[0])
    y0, y1 = int(crop_min[1]), int(crop_max[1])
    x0, x1 = int(crop_min[2]), int(crop_max[2])

    best_response: dict[tuple[int, int, int], float] = {}

    for z in range(z0, z1):
        mask_slice = roi_mask[z]
        roi = mask_slice[y0:y1, x0:x1]
        if not bool(np.any(roi)):
            continue

        mask_u8 = np.zeros_like(mask_slice, dtype=np.uint8)
        mask_u8[y0:y1, x0:x1] = roi.astype(np.uint8) * 255

        img_u8 = extractor._prepare_slice(volume[z])
        keypoints = extractor.sift.detect(img_u8, mask_u8)
        if not keypoints:
            continue

        for kp in keypoints:
            x = int(np.round(kp.pt[0]))
            y = int(np.round(kp.pt[1]))
            if not (y0 <= y < y1 and x0 <= x < x1):
                continue
            if not bool(roi_mask[z, y, x]):
                continue

            key = (z, y, x)
            resp = float(kp.response)
            prev = best_response.get(key)
            if prev is None or resp > prev:
                best_response[key] = resp

    if not best_response:
        return np.zeros((0, 3), dtype=np.int32)

    items = sorted(best_response.items(), key=lambda kv: kv[1], reverse=True)
    items = items[: int(min(n_keypoints, len(items)))]
    return np.asarray([k for k, _ in items], dtype=np.int32)


def _apply_rootsift(descriptors: np.ndarray) -> np.ndarray:
    """
    Apply RootSIFT normalization: L1 normalize -> sqrt -> L2 normalize.

    RootSIFT often improves matching performance by using Hellinger kernel
    instead of Euclidean distance. See: Arandjelovic & Zisserman, 2012.

    Args:
        descriptors: Shape (N, 128) SIFT descriptors

    Returns:
        RootSIFT descriptors, shape (N, 128)
    """
    # L1 normalize
    l1_norm = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
    l1_norm = np.maximum(l1_norm, 1e-8)  # Avoid division by zero
    descriptors = descriptors / l1_norm

    # Element-wise square root
    descriptors = np.sqrt(descriptors)

    # L2 normalize
    l2_norm = np.linalg.norm(descriptors, ord=2, axis=1, keepdims=True)
    l2_norm = np.maximum(l2_norm, 1e-8)
    descriptors = descriptors / l2_norm

    return descriptors


@cloneable
class SIFTDescriptor:
    """
    Stage 11: OpenCV SIFT descriptor extracted on 2D slices.

    For 3D volumes, extracts SIFT from the axial slice at each coordinate.
    Uses dense SIFT (compute at specified locations, not detect).

    Path A from action_plan.md: 2D slice pipeline.

    Note:
        Set compute_orientation=True (default) for rotation-invariant descriptors.
        This computes dominant gradient orientation at each keypoint, matching
        SIFT's design for rotation invariance.

        Set rootsift=True for RootSIFT normalization, which often improves
        matching by using Hellinger kernel. Note: this changes the descriptor
        distribution, so PCA/models trained on standard SIFT won't work well.
    """

    def __init__(
        self,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
        normalize_image: bool = True,
        compute_orientation: bool = True,
        rootsift: bool = False,
        preprocess: str = "none",
    ):
        """
        Args:
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Contrast threshold for filtering weak features
            edge_threshold: Threshold for filtering edge-like features
            sigma: Sigma of Gaussian applied to input image
            normalize_image: Whether to normalize slice to 0-255 before SIFT
            compute_orientation: If True (default), compute dominant orientation
                at each keypoint for rotation invariance. If False, use upright
                orientation (angle=0).
            rootsift: If True, apply RootSIFT normalization (L1 -> sqrt -> L2).
                Note: changes descriptor distribution, requires retraining.
            preprocess: Image preprocessing before descriptor extraction.
                "none" or "phase_congruency".
        """
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV is required for SIFT. Install with: pip install opencv-python")

        self.sift = cv2.SIFT_create(
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self.normalize_image = normalize_image
        self.compute_orientation = compute_orientation
        self.rootsift = rootsift
        self.preprocess = preprocess
        self.dim = 128  # SIFT descriptor dimension

    def _prepare_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Convert slice to uint8 for OpenCV."""
        return prepare_slice_uint8(slice_2d, self.normalize_image, preprocess=self.preprocess)

    def extract(
        self,
        volume: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        """
        Extract SIFT descriptors at 3D coordinates using axial slices.

        Args:
            volume: 3D volume, shape (D, H, W)
            coords: Coordinates, shape (N, 3) as (z, y, x)

        Returns:
            descriptors: Shape (N, 128)
        """
        n = len(coords)
        descriptors = np.zeros((n, self.dim), dtype=np.float32)

        # Group by slice for efficiency
        slice_groups = {}
        for i, (z, y, x) in enumerate(coords):
            z = int(z)
            if z not in slice_groups:
                slice_groups[z] = []
            slice_groups[z].append((i, int(y), int(x)))

        for z, points in slice_groups.items():
            if z < 0 or z >= volume.shape[0]:
                continue

            slice_2d = self._prepare_slice(volume[z])

            # Create keypoints at specified locations
            keypoints = []
            indices = []
            for i, y, x in points:
                if 0 <= x < slice_2d.shape[1] and 0 <= y < slice_2d.shape[0]:
                    # angle=-1 requests orientation computation (if enabled).
                    kp = self.cv2.KeyPoint(float(x), float(y), DEFAULT_KEYPOINT_SIZE, -1)
                    keypoints.append(kp)
                    indices.append(i)

            if not keypoints:
                continue

            # Compute orientations for rotation invariance
            if self.compute_orientation:
                keypoints = _compute_keypoint_orientations(
                    slice_2d, keypoints, self.cv2
                )
            else:
                # Use upright orientation (angle=0)
                keypoints = [self.cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, 0)
                            for kp in keypoints]

            # Compute descriptors at keypoints
            _, descs = self.sift.compute(slice_2d, keypoints)

            if descs is not None:
                for idx, desc in zip(indices, descs):
                    if self.rootsift:
                        # RootSIFT: store OpenCV's L2-normalized descriptor as-is.
                        # _apply_rootsift will apply Hellinger embedding (L1 -> sqrt -> L2).
                        descriptors[idx] = desc
                    else:
                        # Standard SIFT: L2 normalize (redundant but ensures consistency
                        # since OpenCV already returns L2-normalized descriptors)
                        descriptors[idx] = desc / (np.linalg.norm(desc) + 1e-8)

        # Apply RootSIFT normalization if enabled.
        # Per Arandjelovic & Zisserman (2012), RootSIFT takes L2-normalized SIFT
        # (which OpenCV provides) and applies: L1 normalize -> sqrt -> L2 normalize.
        if self.rootsift:
            descriptors = _apply_rootsift(descriptors)

        return descriptors

    def extract_2d(
        self,
        image: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        """
        Extract SIFT descriptors from 2D image.

        Args:
            image: 2D image, shape (H, W)
            coords: Coordinates, shape (N, 2) as (y, x)

        Returns:
            descriptors: Shape (N, 128)
        """
        n = len(coords)
        descriptors = np.zeros((n, self.dim), dtype=np.float32)

        img = self._prepare_slice(image)

        keypoints = []
        valid_indices = []
        for i, (y, x) in enumerate(coords):
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                kp = self.cv2.KeyPoint(float(x), float(y), DEFAULT_KEYPOINT_SIZE, -1)
                keypoints.append(kp)
                valid_indices.append(i)

        if keypoints:
            # Compute orientations for rotation invariance
            if self.compute_orientation:
                keypoints = _compute_keypoint_orientations(img, keypoints, self.cv2)
            else:
                keypoints = [self.cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, 0)
                            for kp in keypoints]

            _, descs = self.sift.compute(img, keypoints)
            if descs is not None:
                for idx, desc in zip(valid_indices, descs):
                    if self.rootsift:
                        # RootSIFT: store OpenCV's L2-normalized descriptor as-is.
                        descriptors[idx] = desc
                    else:
                        # Standard SIFT: L2 normalize
                        descriptors[idx] = desc / (np.linalg.norm(desc) + 1e-8)

        # Apply RootSIFT normalization if enabled.
        # Per Arandjelovic & Zisserman (2012), RootSIFT takes L2-normalized SIFT
        # and applies: L1 normalize -> sqrt -> L2 normalize.
        if self.rootsift:
            descriptors = _apply_rootsift(descriptors)

        return descriptors

    def detect_keypoints_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        max_keypoints: int = 1000,
    ) -> list:
        """Detect SIFT keypoints on a 2D image.

        Args:
            image: 2D grayscale image (H, W)
            mask: Optional uint8 mask (0=invalid, 255=valid). If provided as bool or 0/1, it
                will be converted to uint8 0/255.
            max_keypoints: Cap returned keypoints by top SIFT response

        Returns:
            List of cv2.KeyPoint objects. If compute_orientation=False, keypoint angles are set to 0.
        """
        max_keypoints = int(max_keypoints)
        if max_keypoints <= 0:
            return []

        if image.ndim != 2:
            raise ValueError(f"detect_keypoints_2d expects a 2D image, got shape={image.shape}")

        img = self._prepare_slice(image)

        mask_u8 = None
        if mask is not None:
            if mask.shape != img.shape:
                raise ValueError(
                    f"detect_keypoints_2d mask shape must match image: {mask.shape} != {img.shape}"
                )
            mask_u8 = np.asarray(mask)
            if mask_u8.dtype != np.uint8:
                mask_u8 = mask_u8.astype(np.uint8, copy=False)
            # Convert boolean or 0/1 masks to 0/255.
            if mask_u8.size > 0 and int(mask_u8.max()) <= 1:
                mask_u8 = (mask_u8 * 255).astype(np.uint8, copy=False)

        keypoints = self.sift.detect(img, mask_u8)
        if not keypoints:
            return []

        keypoints = sorted(keypoints, key=lambda kp: float(kp.response), reverse=True)
        keypoints = keypoints[: max_keypoints]

        if not self.compute_orientation:
            keypoints = [
                self.cv2.KeyPoint(
                    float(kp.pt[0]),
                    float(kp.pt[1]),
                    float(kp.size),
                    0.0,
                    float(kp.response),
                    int(getattr(kp, "octave", 0)),
                    int(getattr(kp, "class_id", -1)),
                )
                for kp in keypoints
            ]

        return keypoints

    @staticmethod
    def _keypoint_key(kp) -> tuple[int, int, int, int]:
        """Key for matching cv2.KeyPoint objects across OpenCV compute() calls."""
        # Quantize to avoid float representation differences.
        return (
            int(np.round(float(kp.pt[0]) * 1000.0)),
            int(np.round(float(kp.pt[1]) * 1000.0)),
            int(np.round(float(kp.size) * 1000.0)),
            int(np.round(float(kp.angle) * 1000.0)),
        )

    def compute_2d(
        self,
        image: np.ndarray,
        keypoints: list,
    ) -> np.ndarray:
        """Compute SIFT descriptors at specified keypoints.

        Uses the provided keypoint geometry (x, y, size, angle) as-is. The output descriptor
        order follows the input keypoint order. If OpenCV drops any keypoints, the
        corresponding rows are returned as zeros (and can be filtered downstream).

        Args:
            image: 2D grayscale image (H, W)
            keypoints: List of cv2.KeyPoint objects

        Returns:
            Descriptors of shape (N, 128) aligned by input keypoint order
        """
        if image.ndim != 2:
            raise ValueError(f"compute_2d expects a 2D image, got shape={image.shape}")

        n = len(keypoints)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        img = self._prepare_slice(image)

        desc_out = np.zeros((n, self.dim), dtype=np.float32)

        kps_out, descs = self.sift.compute(img, keypoints)
        if descs is None or not kps_out:
            return desc_out

        descs = descs.astype(np.float32, copy=False)

        if len(kps_out) == n:
            desc_out = descs
        else:
            out_map: dict[tuple[int, int, int, int], np.ndarray] = {}
            for kp, desc in zip(kps_out, descs):
                out_map[self._keypoint_key(kp)] = desc
            for i, kp in enumerate(keypoints):
                desc = out_map.get(self._keypoint_key(kp))
                if desc is not None:
                    desc_out[i] = desc

        if self.rootsift:
            # RootSIFT expects OpenCV-style (already L2-normalized) SIFT as input.
            desc_out = _apply_rootsift(desc_out)
            return desc_out.astype(np.float32, copy=False)

        # Standardize to unit norm for downstream cosine/PCA consistency.
        norms = np.linalg.norm(desc_out, axis=1, keepdims=True)
        desc_out = desc_out / (norms + 1e-8)
        return desc_out.astype(np.float32, copy=False)

    def detect_and_compute_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        max_keypoints: int = 1000,
    ) -> tuple[list, np.ndarray]:
        """Convenience wrapper: detect keypoints then compute descriptors on the same image."""
        keypoints = self.detect_keypoints_2d(image, mask=mask, max_keypoints=max_keypoints)
        descriptors = self.compute_2d(image, keypoints)
        return keypoints, descriptors

    def detect_keypoints_3d(
        self,
        volume: np.ndarray,
        roi_mask: np.ndarray,
        crop_min: np.ndarray,
        crop_max: np.ndarray,
        n_keypoints: int,
    ) -> np.ndarray:
        """Detect SIFT keypoints in a 3D volume within an ROI mask."""
        return _detect_sift_keypoints_3d(
            self, volume, roi_mask, crop_min, crop_max, n_keypoints
        )
