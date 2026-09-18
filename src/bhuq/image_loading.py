"""Load scalar images used by inverse-imaging experiments."""

from pathlib import Path

import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread


def derive_image_name(
    path: str | Path,
    *,
    array_key: str | None = None,
) -> str:
    """
    Derive a stable image name from its file or selected NPZ array.

    Parameters:
    -----------
    path: str | Path
        Source image path.
    array_key: str | None
        Selected NPZ lookup key, which takes precedence over the filename.

    Returns:
    --------
    str
        Single path-safe image name.

    Raises:
    -------
    ValueError
        If the derived name is empty, contains whitespace, or is not one path
        segment.
    """
    name = array_key or Path(path).stem
    if (
        not name
        or any(character.isspace() for character in name)
        or Path(name).name != name
    ):
        raise ValueError(
            "Image filename stem or NPZ array key must be one non-empty "
            "path segment without whitespace."
        )
    return name


def load_scalar_image(
    path: str | Path,
    *,
    array_key: str | None = None,
) -> np.ndarray:
    """
    Load a finite two-dimensional scalar image.

    Parameters:
    -----------
    path: str | Path
        ehtim text image, NPY, NPZ, or standard image file.
    array_key: str | None
        NPZ lookup key. It may be omitted when the archive contains one array.

    Returns:
    --------
    np.ndarray
        Floating-point image with shape `(height, width)`.

    Raises:
    -------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the format is ambiguous or does not contain a finite scalar image.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    suffix = image_path.suffix.lower()
    if suffix == ".txt":
        import ehtim as eh

        image = np.asarray(
            eh.image.load_txt(str(image_path)).imarr(),
            dtype=np.float32,
        )
    elif suffix == ".npy":
        image = np.asarray(
            np.load(image_path, allow_pickle=False),
            dtype=np.float32,
        )
    elif suffix == ".npz":
        image = _load_npz_image(image_path, array_key=array_key)
    else:
        image = _load_standard_image(image_path)

    if image.ndim != 2 or image.size == 0:
        raise ValueError(
            f"Scalar image must be a non-empty two-dimensional array: "
            f"{image_path}."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Scalar image contains non-finite values: {image_path}.")
    return image


def normalize_image_by_maximum(image: np.ndarray) -> np.ndarray:
    """
    Scale a finite scalar image so its maximum intensity is one.

    Parameters:
    -----------
    image: np.ndarray
        Non-empty image with a positive maximum intensity.

    Returns:
    --------
    np.ndarray
        Floating-point image divided by its maximum.
    """
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0 or not np.all(np.isfinite(image_array)):
        raise ValueError("`image` must contain finite values.")
    maximum_intensity = float(image_array.max())
    if maximum_intensity <= 0.0:
        raise ValueError("`image` must have a positive maximum intensity.")
    return image_array / maximum_intensity


def _load_npz_image(
    path: Path,
    *,
    array_key: str | None,
) -> np.ndarray:
    """Load one explicitly selected image from an NPZ archive."""
    with np.load(path, allow_pickle=False) as archive:
        if array_key is None:
            if len(archive.files) != 1:
                raise ValueError(
                    f"`array_key` is required for NPZ archive with "
                    f"{len(archive.files)} arrays: {path}."
                )
            selected_name = archive.files[0]
        else:
            selected_name = array_key
        if selected_name not in archive.files:
            raise ValueError(
                f"NPZ archive does not contain `{selected_name}`: {path}."
            )
        return np.asarray(archive[selected_name], dtype=np.float32)


def _load_standard_image(path: Path) -> np.ndarray:
    """Load a grayscale, RGB, or RGBA image as scalar intensity."""
    image = np.asarray(imread(path))
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}.")
    if image.shape[-1] == 4:
        image = rgba2rgb(image)
    if image.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}.")
    return np.asarray(rgb2gray(image), dtype=np.float32)
