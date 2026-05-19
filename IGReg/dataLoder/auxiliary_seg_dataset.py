import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# If this file is imported from the project root, use dataLoder.dataload_utils.
# If this file is run inside the dataLoder folder, fall back to dataload_utils.
try:
    from dataLoder.dataload_utils import load_pickle, default_loader, default_loader_test
except ImportError:
    from dataload_utils import load_pickle, default_loader, default_loader_test

# =========================
# Path configuration
# =========================
TITLE_DIR = Path("/mnt/Chichien")

ORIGINAL_IMG_DIR = TITLE_DIR / "DATA_ALL/DATA/MICCAI_2021"

EXCLUDE_NAME_PATHS = [
    TITLE_DIR / "DATA_ALL/name_20_use_in_21.npy",
    TITLE_DIR / "DATA_ALL/name_UCSF_use_in_21.npy",
]

KFOLD_PATH = TITLE_DIR / "DATA_ALL/k_fold_Multi_final/splits_MI21_2024.pkl"


def convert_aux_seg_to_multilabel(seg_in: np.ndarray) -> np.ndarray:
    """
    Convert the original MICCAI/BraTS-style segmentation mask into
    three nested binary segmentation targets.

    Output channels:
        0: whole tumor region
        1: tumor core region
        2: enhancing tumor region
    """
    seg = np.zeros_like(seg_in)

    # Remap original labels to an ordered auxiliary label space.
    seg[seg_in == 2] = 1
    seg[seg_in == 1] = 2
    seg[seg_in == 4] = 3

    whole_tumor = (seg >= 1).astype(np.float32)
    tumor_core = (seg >= 2).astype(np.float32)
    enhancing_tumor = (seg == 3).astype(np.float32)
    seg_out = np.stack(
        [whole_tumor, tumor_core, enhancing_tumor],
        axis=0
    )
    return seg_out


def load_excluded_names(name_paths: Sequence[Union[str, Path]]) -> set:
    """
    Load case names that should be excluded from the auxiliary dataset.
    These cases have been used by the main datasets.
    """
    excluded_names = []

    for name_path in name_paths:
        names = np.load(name_path, allow_pickle=True)
        excluded_names.extend([str(name) for name in names])

    return set(excluded_names)


def build_auxiliary_case_list(
    split_names: Sequence[str],
    image_root: Union[str, Path],
    excluded_names: set,
) -> Tuple[List[str], List[str]]:
    """
    Build the final auxiliary dataset case list after excluding cases
    already used by the main datasets.
    """
    image_root = Path(image_root)

    case_names = []
    case_paths = []

    for case_name in split_names:
        case_name = str(case_name)

        if case_name in excluded_names:
            continue

        case_names.append(case_name)
        case_paths.append(str(image_root / case_name))

    return case_names, case_paths


class AuxiliarySegDataset(Dataset):
    """
    Auxiliary segmentation dataset.

    This dataset is used only as an auxiliary dataset. It excludes cases
    that have already been used in the main datasets and returns only
    T1C/T2 images with segmentation targets.
    """

    def __init__(
        self,
        split: str = "train",
        cross: int = 1,
        transform=None,
        image_root: Union[str, Path] = ORIGINAL_IMG_DIR,
        kfold_path: Union[str, Path] = KFOLD_PATH,
        exclude_name_paths: Sequence[Union[str, Path]] = EXCLUDE_NAME_PATHS,
        input_modalities: Tuple[int, int] = (0, 1),
        return_name: bool = False,
    ):
        """
        Args:
            split: 'train' or 'val'.
            cross: fold index, starting from 1.
            transform: transform applied to both image and segmentation mask.
            image_root: root directory of MICCAI_2021 images.
            kfold_path: path to the k-fold split file.
            exclude_name_paths: paths of case-name lists to be excluded.
            input_modalities: selected image modality indices. Default uses T1C and T2.
            return_name: whether to return case name and path.
        """
        super().__init__()

        assert split in ["train", "val"], "split must be 'train' or 'val'."

        self.split = split
        self.cross = cross
        self.transform = transform
        self.input_modalities = input_modalities
        self.return_name = return_name

        splits = load_pickle(str(kfold_path))
        split_names = splits[cross - 1][split]

        excluded_names = load_excluded_names(exclude_name_paths)

        self.case_names, self.case_paths = build_auxiliary_case_list(
            split_names=split_names,
            image_root=image_root,
            excluded_names=excluded_names,
        )

        # Use random crop / augmentation loader for training,
        # and deterministic loader for validation.
        self.loader = default_loader if split == "train" else default_loader_test

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        case_name = self.case_names[idx]

        imgs, mask = self.loader(case_path)

        # mask[0] is usually the brain/background mask.
        # mask[1] is the tumor segmentation mask.
        seg_target = convert_aux_seg_to_multilabel(mask[1])

        if self.transform is not None:
            imgs = self.transform(imgs)
            seg_target = self.transform(seg_target)

        # Select auxiliary input modalities, e.g., T1C and T2.
        imgs = torch.stack(
            [imgs[m] for m in self.input_modalities],
            dim=0
        ).float()

        seg_target = seg_target.float()

        if self.return_name:
            return imgs, seg_target, case_name, case_path

        return imgs, seg_target


# =========================
# Optional split generation
# =========================
def write_pickle(obj, file: Union[str, Path], mode: str = "wb") -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def save_kfold_split(
    all_case_names: Sequence[str],
    splits_file: Union[str, Path],
    n_splits: int = 5,
    seed: int = 12345,
) -> None:
    """
    Save k-fold splits for the auxiliary dataset.
    """
    from collections import OrderedDict
    from sklearn.model_selection import KFold

    all_case_names = np.sort(np.array(all_case_names))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for train_idx, val_idx in kfold.split(all_case_names):
        split_dict = OrderedDict()
        split_dict["train"] = all_case_names[train_idx]
        split_dict["val"] = all_case_names[val_idx]
        splits.append(split_dict)

    write_pickle(splits, splits_file)


if __name__ == "__main__":
    import monai
    from torch.utils.data import DataLoader

    transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),
    ])

    train_dataset = AuxiliarySegDataset(
        split="train",
        cross=1,
        transform=transforms,
        return_name=False,
    )

    val_dataset = AuxiliarySegDataset(
        split="val",
        cross=1,
        transform=transforms,
        return_name=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Auxiliary training cases: {len(train_dataset)}")
    print(f"Auxiliary validation cases: {len(val_dataset)}")

    for imgs, seg in train_loader:
        print(imgs.shape, seg.shape)
        break