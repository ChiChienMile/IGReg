import csv
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np


# =========================
# Basic file utilities
# =========================
def load_pickle(file: Union[str, Path], mode: str = "rb"):
    """
    Load a pickle file.

    This is used to load the k-fold split files.
    """
    with open(file, mode) as f:
        return pickle.load(f)


def read_csv_rows(file: Union[str, Path]) -> List[List[str]]:
    """
    Read a CSV file as a list of rows.

    The first row is usually the header.
    """
    with open(file, mode="r", encoding="utf-8") as f:
        return list(csv.reader(f))


def find_value_by_name(
    name: str,
    info_file: Union[str, Path],
    name_col: int,
    value_col: int,
    strip_mr_prefix: bool = False,
):
    """
    Find a clinical value according to the case name.

    Args:
        name: case name.
        info_file: clinical CSV file.
        name_col: column index of case names.
        value_col: column index of the target label.
        strip_mr_prefix: whether to remove the 'MR_' prefix before matching.

    Returns:
        The matched raw value. If no matched case is found, returns None.
    """
    if strip_mr_prefix:
        name = name.replace("MR_", "")

    rows = read_csv_rows(info_file)

    for row in rows[1:]:
        if len(row) <= max(name_col, value_col):
            continue

        if row[name_col] == name:
            return row[value_col]

    return None


# =========================
# Label readers: MICCAI_2020
# =========================
def get_label_20HL(name: str, info_file: Union[str, Path]) -> int:
    """
    Read low-/high-grade glioma label from the MICCAI_2020 clinical file.

    Label definition:
        LGG -> 0
        HGG -> 1
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=1,
        value_col=0,
    )

    if value == "LGG":
        return 0
    if value == "HGG":
        return 1

    return -1


def get_label_20IDH(name: str, info_file: Union[str, Path]) -> int:
    """
    Read IDH mutation label from the MICCAI_2020 clinical file.

    Label definition:
        WT     -> 0
        Mutant -> 1
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=1,
        value_col=6,
    )

    if value == "WT":
        return 0
    if value == "Mutant":
        return 1

    return -1


def get_label_201p19q(name: str, info_file: Union[str, Path]) -> int:
    """
    Read 1p/19q co-deletion label from the MICCAI_2020 clinical file.

    Label definition:
        non-codel -> 0
        codel     -> 1
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=1,
        value_col=7,
    )

    if value == "non-codel":
        return 0
    if value == "codel":
        return 1

    return -1


# =========================
# Label readers: BMIAXNAT
# =========================
def get_label_BMIAXNAT_IDH(name: str, info_file: Union[str, Path]) -> int:
    """
    Read IDH mutation label from the BMIAXNAT clinical file.

    The case name in the clinical file does not contain the 'MR_' prefix.
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=0,
        value_col=1,
        strip_mr_prefix=True,
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def get_label_BMIAXNAT_1p19q(name: str, info_file: Union[str, Path]) -> int:
    """
    Read 1p/19q co-deletion label from the BMIAXNAT clinical file.

    The case name in the clinical file does not contain the 'MR_' prefix.
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=0,
        value_col=2,
        strip_mr_prefix=True,
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def get_label_BMIAXNAT_HL(name: str, info_file: Union[str, Path]) -> int:
    """
    Read low-/high-grade glioma label from the BMIAXNAT clinical file.

    Original grade definition:
        grade 2 or 3 -> low-grade label 0
        grade 4      -> high-grade label 1
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=0,
        value_col=3,
        strip_mr_prefix=True,
    )

    try:
        grade = int(value)
    except (TypeError, ValueError):
        return -1

    if grade == 4:
        return 1
    if grade in [2, 3]:
        return 0

    return -1


# =========================
# Label reader: public 1p19q dataset
# =========================
def get_label_1p19q(name: str, info_file: Union[str, Path]) -> int:
    """
    Read 1p/19q co-deletion label from the public 1p19q dataset.

    Label definition:
        0 -> non-codeletion
        1 -> codeletion
    """
    value = find_value_by_name(
        name=name,
        info_file=info_file,
        name_col=0,
        value_col=4,
    )

    try:
        label = int(value)
    except (TypeError, ValueError):
        return -1

    if label in [0, 1]:
        return label

    return -1


# =========================
# NIfTI image utilities
# =========================
def load_nii(filename: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file.

    Returns:
        data: image array.
        affine: affine matrix.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filename}")

    nii = nib.load(str(filename))
    data = nii.get_fdata()
    affine = nii.affine
    nii.uncache()

    return data, affine


def load_mask(case_dir: Union[str, Path]) -> np.ndarray:
    """
    Load the tumor mask from a case directory.

    Expected file:
        mask.nii
    """
    case_dir = Path(case_dir)
    mask, _ = load_nii(case_dir / "mask.nii")

    return np.asarray(mask, dtype=np.float32)


def load_img(case_dir: Union[str, Path], modality: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and normalize one MRI modality.

    Expected files:
        T1C.nii
        T2.nii

    Normalization:
        z-score normalization is applied only within the nonzero foreground region.

    Returns:
        img: normalized image.
        foreground_mask: binary foreground mask.
        affine: affine matrix.
    """
    case_dir = Path(case_dir)
    img, affine = load_nii(case_dir / f"{modality}.nii")

    img = np.asarray(img, dtype=np.float32)
    foreground_mask = img > 0

    eps = 1e-8
    if np.any(foreground_mask):
        img[foreground_mask] = (
            img[foreground_mask] - img[foreground_mask].mean()
        ) / (img[foreground_mask].std() + eps)

    foreground_mask = foreground_mask.astype(np.float32)

    return img, foreground_mask, affine


# =========================
# Crop and padding utilities
# =========================
def get_foreground_bbox(mask: np.ndarray):
    """
    Get the bounding box of a binary foreground mask.

    Returns:
        min_x, max_x, min_y, max_y, min_z, max_z

    If the mask is empty, returns None.
    """
    foreground = np.argwhere(mask > 0)

    if foreground.size == 0:
        return None

    min_x, min_y, min_z = foreground.min(axis=0)
    max_x, max_y, max_z = foreground.max(axis=0)

    return min_x, max_x, min_y, max_y, min_z, max_z


def center_pad_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Center-pad a 3D array to the target shape.

    This function assumes that each dimension of data is not larger than target_shape.
    """
    data = np.asarray(data, dtype=np.float32)

    pad_width = []

    for current_size, target_size in zip(data.shape, target_shape):
        if current_size > target_size:
            raise ValueError(
                f"Cannot pad array with shape {data.shape} to smaller target shape {target_shape}."
            )

        total_pad = target_size - current_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))

    return np.pad(data, pad_width=pad_width, mode="constant", constant_values=0)


def random_shift_crop_with_tumor(
    img_t1c: np.ndarray,
    img_t2: np.ndarray,
    mask_t1c: np.ndarray,
    mask_t2: np.ndarray,
    tumor_mask: np.ndarray,
    crop_prob: float = 0.8,
):
    """
    Apply a simple tumor-preserving random spatial shift.

    The original implementation used a random crop followed by center padding.
    Since the crop size is the same as the original image size, this operation
    mainly shifts the image content while keeping the tumor region inside the crop.

    With probability 1 - crop_prob, the original arrays are returned.
    """
    if np.random.random() >= crop_prob:
        return img_t1c, img_t2, mask_t1c, mask_t2, tumor_mask

    bbox = get_foreground_bbox(tumor_mask)

    if bbox is None:
        return img_t1c, img_t2, mask_t1c, mask_t2, tumor_mask

    min_x, _, min_y, _, min_z, _ = bbox
    target_shape = img_t1c.shape

    # Choose crop start positions before the tumor bounding box.
    # This keeps the tumor region inside the valid crop.
    start_x = np.random.randint(0, int(min_x) + 1) if min_x > 0 else 0
    start_y = np.random.randint(0, int(min_y) + 1) if min_y > 0 else 0
    start_z = np.random.randint(0, int(min_z) + 1) if min_z > 0 else 0

    def crop_and_pad(arr: np.ndarray) -> np.ndarray:
        cropped = arr[
            start_x:target_shape[0],
            start_y:target_shape[1],
            start_z:target_shape[2],
        ]
        return center_pad_to_shape(cropped, target_shape)

    img_t1c = crop_and_pad(img_t1c)
    img_t2 = crop_and_pad(img_t2)
    mask_t1c = crop_and_pad(mask_t1c)
    mask_t2 = crop_and_pad(mask_t2)
    tumor_mask = crop_and_pad(tumor_mask)

    return img_t1c, img_t2, mask_t1c, mask_t2, tumor_mask


# =========================
# Loader construction
# =========================
def build_image_and_masks(
    img_t1c: np.ndarray,
    img_t2: np.ndarray,
    mask_t1c: np.ndarray,
    mask_t2: np.ndarray,
    tumor_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build image and mask arrays returned by the dataloader.

    Returned images:
        imgs[0]: T1C image.
        imgs[1]: T2 image.

    Returned masks:
        masks[0]: brain/foreground mask.
        masks[1]: tumor-region mask.
        masks[2]: non-tumor foreground mask.
        masks[3]: binary tumor mask.
    """
    brain_mask = ((mask_t1c + mask_t2) > 0).astype(np.float32)
    tumor_mask = (tumor_mask > 0).astype(np.float32)

    tumor_region_mask = tumor_mask * brain_mask
    non_tumor_mask = (1.0 - tumor_mask) * brain_mask

    imgs = np.stack([img_t1c, img_t2], axis=0).astype(np.float32)
    masks = np.stack(
        [
            brain_mask,
            tumor_region_mask,
            non_tumor_mask,
            tumor_mask,
        ],
        axis=0,
    ).astype(np.float32)

    return imgs, masks


def default_loader(case_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Training loader for the classification datasets.

    It loads T1C, T2, and tumor mask, applies a tumor-preserving random shift,
    and returns image/mask arrays.

    Returns:
        imgs: shape [2, D, H, W]
        masks: shape [4, D, H, W]
    """
    img_t1c, mask_t1c, _ = load_img(case_dir, "T1C")
    img_t2, mask_t2, _ = load_img(case_dir, "T2")

    tumor_mask = load_mask(case_dir)

    img_t1c, img_t2, mask_t1c, mask_t2, tumor_mask = random_shift_crop_with_tumor(
        img_t1c=img_t1c,
        img_t2=img_t2,
        mask_t1c=mask_t1c,
        mask_t2=mask_t2,
        tumor_mask=tumor_mask,
        crop_prob=0.8,
    )

    imgs, masks = build_image_and_masks(
        img_t1c=img_t1c,
        img_t2=img_t2,
        mask_t1c=mask_t1c,
        mask_t2=mask_t2,
        tumor_mask=tumor_mask,
    )

    return imgs, masks


def default_loader_test(case_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic loader for validation, testing, or prediction.

    No random crop or spatial shift is applied.

    Returns:
        imgs: shape [2, D, H, W]
        masks: shape [4, D, H, W]
    """
    img_t1c, mask_t1c, _ = load_img(case_dir, "T1C")
    img_t2, mask_t2, _ = load_img(case_dir, "T2")

    tumor_mask = load_mask(case_dir)

    imgs, masks = build_image_and_masks(
        img_t1c=img_t1c,
        img_t2=img_t2,
        mask_t1c=mask_t1c,
        mask_t2=mask_t2,
        tumor_mask=tumor_mask,
    )

    return imgs, masks