import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# If this file is imported from the project root, use dataLoder.dataload_utils.
# If this file is run inside the dataLoder folder, fall back to dataload_utils.
try:
    from dataLoder.dataload_utils import (
        default_loader,
        default_loader_test,
        load_pickle,
        get_label_BMIAXNAT_IDH,
        get_label_20IDH,
    )
except ImportError:
    from dataload_utils import (
        default_loader,
        default_loader_test,
        load_pickle,
        get_label_BMIAXNAT_IDH,
        get_label_20IDH,
    )


# =========================
# Global path configuration
# =========================
TITLE_DIR = Path("/mnt/Chichien")
# Poor image quality
EXCLUDED_CASES = {"MR_EGD-0379", "MR_EGD-0611"}

KFOLD_NAME_0 = "splits_final_0_IDH.pkl"
KFOLD_NAME_1 = "splits_final_1_IDH.pkl"


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for each dataset used in the IDH prediction task.
    """
    name: str
    image_root: Path
    split_file_0: Path
    split_file_1: Path
    label_file: Path
    label_getter: Callable[[str, str], int]


@dataclass(frozen=True)
class CaseRecord:
    """
    A single labeled case record.
    """
    name: str
    path: str
    label: int
    source: str


DATASET_CONFIGS = [
    DatasetConfig(
        name="BMIAXNAT",
        image_root=TITLE_DIR / "DATA_ALL/DATA/BMIAXNAT_after_registration_padding",
        split_file_0=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/BMIAXNAT" / KFOLD_NAME_0,
        split_file_1=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/BMIAXNAT" / KFOLD_NAME_1,
        label_file=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/Clinical/labels_BMIAXNAT.csv",
        label_getter=get_label_BMIAXNAT_IDH,
    ),
    DatasetConfig(
        name="MICCAI_2020",
        image_root=TITLE_DIR / "DATA_ALL/DATA/MICCAI_2020_IDH_TCIA_T",
        split_file_0=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/MI21" / KFOLD_NAME_0,
        split_file_1=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/MI21" / KFOLD_NAME_1,
        label_file=TITLE_DIR / "DATA_ALL/k_fold_Multi_final/Clinical/labels_MICCAI_2020_IDH.csv",
        label_getter=get_label_20IDH,
    ),
]


def normalize_split_name(read_type: str) -> str:
    """
    Keep compatibility with the original code.
    Only 'train' uses the training split, and all other values use the validation split.
    """
    return "train" if read_type == "train" else "val"


def load_split_names(
    split_file_0: Union[str, Path],
    split_file_1: Union[str, Path],
    read_type: str,
    cross: int,
) -> List[str]:
    """
    Load case names from two class-specific split files and merge them.

    Cross is 1-based:
        Cross=1 means the first fold.
    """
    if cross < 1:
        raise ValueError("Cross should be 1-based. Use Cross=1 for the first fold.")

    split_name = normalize_split_name(read_type)

    splits_0 = load_pickle(str(split_file_0))
    splits_1 = load_pickle(str(split_file_1))

    names_0 = splits_0[cross - 1][split_name]
    names_1 = splits_1[cross - 1][split_name]

    return [str(x) for x in np.concatenate((names_0, names_1), axis=0)]


def collect_records_from_dataset(
    config: DatasetConfig,
    read_type: str,
    cross: int,
) -> List[CaseRecord]:
    """
    Collect valid labeled cases from one dataset.

    Cases with invalid labels or manually excluded case names are skipped.
    """
    case_names = load_split_names(
        split_file_0=config.split_file_0,
        split_file_1=config.split_file_1,
        read_type=read_type,
        cross=cross,
    )

    records = []

    for case_name in case_names:
        if case_name in EXCLUDED_CASES:
            continue

        label = int(config.label_getter(case_name, str(config.label_file)))

        if label not in [0, 1]:
            continue

        records.append(
            CaseRecord(
                name=case_name,
                path=str(config.image_root / case_name),
                label=label,
                source=config.name,
            )
        )

    return records


def collect_all_records(read_type: str, cross: int) -> List[CaseRecord]:
    """
    Collect all valid IDH cases from BMIAXNAT and MICCAI_2020.
    """
    all_records = []

    for config in DATASET_CONFIGS:
        all_records.extend(
            collect_records_from_dataset(
                config=config,
                read_type=read_type,
                cross=cross,
            )
        )

    return all_records


def allocate_counts(total: int, group_sizes: Sequence[int]) -> List[int]:
    """
    Allocate a fixed number of samples according to group proportions.

    This follows the original proportional allocation idea, but additionally
    fills the remaining samples caused by integer truncation.
    """
    group_sizes = list(group_sizes)
    available_total = sum(group_sizes)

    if total <= 0 or available_total == 0:
        return [0 for _ in group_sizes]

    total = min(int(total), available_total)

    raw_counts = [
        total * size / available_total if available_total > 0 else 0
        for size in group_sizes
    ]

    counts = [
        min(int(raw), size)
        for raw, size in zip(raw_counts, group_sizes)
    ]

    remaining = total - sum(counts)

    # Fill the remaining samples according to the largest fractional parts.
    order = sorted(
        range(len(group_sizes)),
        key=lambda i: raw_counts[i] - int(raw_counts[i]),
        reverse=True,
    )

    while remaining > 0:
        updated = False

        for i in order:
            if counts[i] < group_sizes[i]:
                counts[i] += 1
                remaining -= 1
                updated = True

                if remaining == 0:
                    break

        if not updated:
            break

    return counts


def build_model_input(
    imgs,
    masks,
    transform=None,
    input_modalities: Tuple[int, int] = (0, 1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build model input from selected MRI modalities.

    masks[0]: brain/background mask, used to suppress irrelevant background.
    masks[1]: tumor mask, optionally used for tumor masking or concatenation.
    """
    if transform is not None:
        imgs = transform(imgs)
        masks = transform(masks)

    imgs = torch.as_tensor(imgs).float()
    masks = torch.as_tensor(masks).float()

    brain_mask = masks[0]
    tumor_mask = masks[1]

    selected_imgs = []

    for modality_id in input_modalities:
        selected_imgs.append(imgs[modality_id] * brain_mask)

    input_img = torch.stack(selected_imgs, dim=0)

    return input_img, tumor_mask


class Dataset_TrainBal_Ratio(Dataset):
    """
    Ratio-controlled balanced training dataset for IDH prediction.

    In our setting:
        basic_len = 48
        basic_ratio in {1, 5, 10}

    Therefore, the target auxiliary dataset scale is:
        ratio = 1  -> 48 cases
        ratio = 5  -> 240 cases
        ratio = 10 -> 480 cases

    Each __getitem__ returns one label-0 case and one label-1 case:
        imgs:   [2, C, D, H, W]
        labels: [2]
    """

    def __init__(
        self,
        basic_len: int = 48,
        basic_ratio: int = 1,
        transform=None,
        read_type: str = "train",
        Cross: int = 1,
        input_modalities: Tuple[int, int] = (0, 1),
    ):
        self.transform = transform
        self.read_type = read_type
        self.cross = Cross
        self.input_modalities = input_modalities

        if basic_len != 48:
            raise ValueError("For this setting, basic_len should be fixed to 48.")

        if basic_ratio not in [1, 5, 10]:
            raise ValueError("basic_ratio should be one of [1, 5, 10].")

        all_records = collect_all_records(read_type=read_type, cross=Cross)

        label_0_records = [r for r in all_records if r.label == 0]
        label_1_records = [r for r in all_records if r.label == 1]

        if len(label_0_records) == 0 or len(label_1_records) == 0:
            raise RuntimeError(
                f"Both classes are required, but got "
                f"label-0={len(label_0_records)}, label-1={len(label_1_records)}."
            )

        target_total = int(basic_len * basic_ratio)

        # First allocate samples according to the original class ratio.
        num_label_0, num_label_1 = allocate_counts(
            total=target_total,
            group_sizes=[len(label_0_records), len(label_1_records)],
        )

        # Then allocate samples according to the original dataset-source ratio.
        label_0_by_source = [
            [r for r in label_0_records if r.source == config.name]
            for config in DATASET_CONFIGS
        ]
        label_1_by_source = [
            [r for r in label_1_records if r.source == config.name]
            for config in DATASET_CONFIGS
        ]

        label_0_source_counts = allocate_counts(
            total=num_label_0,
            group_sizes=[len(x) for x in label_0_by_source],
        )
        label_1_source_counts = allocate_counts(
            total=num_label_1,
            group_sizes=[len(x) for x in label_1_by_source],
        )

        self.label_0_records = []
        self.label_1_records = []

        # Keep the original deterministic slicing behavior.
        for source_records, source_count in zip(label_0_by_source, label_0_source_counts):
            self.label_0_records.extend(source_records[:source_count])

        for source_records, source_count in zip(label_1_by_source, label_1_source_counts):
            self.label_1_records.extend(source_records[:source_count])

        self.records = self.label_0_records + self.label_1_records

        print(
            f"[IDH Ratio Dataset] "
            f"read_type={read_type}, Cross={Cross}, "
            f"basic_len={basic_len}, ratio={basic_ratio}, "
            f"target_total={target_total}, "
            f"selected label-0={len(self.label_0_records)}, "
            f"selected label-1={len(self.label_1_records)}, "
            f"total={len(self.records)}"
        )

    def __len__(self):
        # Keep the ratio-controlled epoch length.
        return len(self.records)

    def _load_one_case(self, record: CaseRecord) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one training case using the random-crop training loader.
        """
        imgs, masks = default_loader(record.path)

        input_img, _ = build_model_input(
            imgs=imgs,
            masks=masks,
            transform=self.transform,
            input_modalities=self.input_modalities,
        )

        label = torch.tensor(record.label, dtype=torch.long)

        return input_img, label

    def __getitem__(self, idx):
        # Sample one label-0 case.
        index_0 = idx % len(self.label_0_records)

        # Sample one label-1 case with a random shift to avoid fixed pairs.
        random_shift = random.randint(0, len(self.label_1_records) - 1)
        index_1 = (idx + random_shift) % len(self.label_1_records)

        img_0, label_0 = self._load_one_case(self.label_0_records[index_0])
        img_1, label_1 = self._load_one_case(self.label_1_records[index_1])

        imgs = torch.stack((img_0, img_1), dim=0)
        labels = torch.stack((label_0, label_1), dim=0)

        return imgs.float(), labels.long()


class Dataset_Test(Dataset):
    """
    Deterministic dataset for validation, testing, or prediction.

    It returns one case at a time:
        img:   [C, D, H, W]
        label: scalar tensor
    """

    def __init__(
        self,
        transform=None,
        read_type: str = "train",
        Cross: int = 1,
        input_modalities: Tuple[int, int] = (0, 1),
    ):
        self.transform = transform
        self.read_type = read_type
        self.cross = Cross
        self.input_modalities = input_modalities

        self.records = collect_all_records(read_type=read_type, cross=Cross)

        print(
            f"[IDH Test Dataset] "
            f"read_type={read_type}, Cross={Cross}, "
            f"total={len(self.records)}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        imgs, masks = default_loader_test(record.path)

        input_img, tumor_mask = build_model_input(
            imgs=imgs,
            masks=masks,
            transform=self.transform,
            input_modalities=self.input_modalities,
        )

        label = torch.tensor(record.label, dtype=torch.long)

        return input_img.float(), label.long()


if __name__ == "__main__":
    import monai
    from torch.utils.data import DataLoader

    transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),
    ])

    train_dataset = Dataset_TrainBal_Ratio(
        basic_len=48,
        basic_ratio=1,
        transform=transforms,
        read_type="train",
        Cross=1,
    )

    test_dataset = Dataset_Test(
        transform=transforms,
        read_type="val",
        Cross=1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Testing dataset length: {len(test_dataset)}")

    for imgs, labels in train_loader:
        print("Train imgs:", imgs.shape)
        print("Train labels:", labels.shape)
        break

    for imgs, labels in test_loader:
        print("Test imgs:", imgs.shape)
        print("Test labels:", labels.shape)
        break