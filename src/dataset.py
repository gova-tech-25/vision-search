import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.config import (
    DATASET_DIR,
    SPLIT_INFO_PATH,
    TEST_SPLIT_SIZE,
    VAL_SPLIT_SIZE,
)
from PIL import Image


class Flickr30kDataset:
    """
    Flickr30k Dataset loader and validator.

    Preserves the 1-to-many relationship:
        image_id -> image_path -> list[captions]

    Week 1 responsibilities:
    - Dataset discovery
    - Image-caption mapping
    - Missing-image detection
    - Empty-caption detection
    - Image integrity validation
    - Dataset statistics
    - Deterministic train/validation/test splits

    The original dataset is never modified.
    """

    EXPECTED_IMAGES = 31783
    EXPECTED_CAPTIONS = 158915
    EXPECTED_CAPTIONS_PER_IMAGE = 5

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR

        self._validate_dataset_dir()

        self.images_dir = self._find_images_dir()
        self.captions_file = self._find_captions_file()

        # image_id -> {
        #     "image_id": str,
        #     "image_path": str,
        #     "captions": List[str],
        #     "exists": bool
        # }
        self.image_data: Dict[str, Dict] = {}

        # Raw annotation quality information
        self.total_caption_rows = 0
        self.empty_caption_count = 0
        self.empty_caption_records: List[Dict] = []

        self.load_dataset()

        # Dataset splits
        self.splits: Dict[str, List[str]] = {}
        self.setup_splits()

    # ==========================================================
    # DATASET DISCOVERY
    # ==========================================================

    def _validate_dataset_dir(self):
        """Verify that the dataset directory exists."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Flickr30k dataset directory not found at: "
                f"'{self.dataset_dir}'\n"
                f"Expected structure:\n"
                f"  {self.dataset_dir}/\n"
                f"  ├── Images/\n"
                f"  └── captions.txt\n"
            )

    def _find_images_dir(self) -> Path:
        """Find the directory containing Flickr30k images."""
        candidates = [
            self.dataset_dir / "Images",
            self.dataset_dir / "images",
            self.dataset_dir / "flickr30k_images",
            self.dataset_dir,
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                jpgs = list(candidate.glob("*.jpg"))

                if jpgs:
                    return candidate.resolve()

        raise FileNotFoundError(
            f"Could not locate image directory with .jpg files under "
            f"'{self.dataset_dir}'."
        )

    def _find_captions_file(self) -> Path:
        """Find the caption annotation file."""
        candidates = [
            self.dataset_dir / "captions.txt",
            self.dataset_dir / "results.csv",
            self.dataset_dir / "captions.csv",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not locate captions annotation file under "
            f"'{self.dataset_dir}'."
        )

    # ==========================================================
    # DATASET LOADING
    # ==========================================================

    def load_dataset(self):
        """
        Load and parse the image-caption mapping.

        Empty captions are NOT silently ignored anymore.
        They are recorded as dataset-quality anomalies.
        """

        df = pd.read_csv(self.captions_file)

        self.total_caption_rows = len(df)

        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Identify image column
        img_col = next(
            (
                c
                for c in df.columns
                if "img" in c or "image" in c or "file" in c
            ),
            df.columns[0],
        )

        # Identify caption column
        cap_col = next(
            (
                c
                for c in df.columns
                if "cap" in c or "comment" in c
            ),
            df.columns[1],
        )

        print(
            f"Loading captions from '{self.captions_file.name}' "
            f"using columns: ({img_col}, {cap_col})..."
        )

        for row_number, (_, row) in enumerate(df.iterrows(), start=2):

            raw_img_id = str(row[img_col]).strip()

            # Detect missing/empty caption
            if pd.isna(row[cap_col]):
                caption = ""
            else:
                caption = str(row[cap_col]).strip()

            image_id = (
                raw_img_id
                if raw_img_id.lower().endswith(".jpg")
                else f"{raw_img_id}.jpg"
            )

            img_path = self.images_dir / image_id

            # Initialize image record
            if image_id not in self.image_data:
                self.image_data[image_id] = {
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "captions": [],
                    "exists": img_path.exists(),
                }

            # Record empty caption as an anomaly
            if not caption:
                self.empty_caption_count += 1

                self.empty_caption_records.append(
                    {
                        "row": row_number,
                        "image_id": image_id,
                        "caption": "",
                    }
                )

                continue

            # Store valid caption
            self.image_data[image_id]["captions"].append(caption)

        # Report missing images
        missing_count = sum(
            1
            for data in self.image_data.values()
            if not data["exists"]
        )

        if missing_count > 0:
            print(
                f"Warning: {missing_count} referenced images were not "
                f"found in '{self.images_dir}'."
            )

        # Only valid image files remain in image_data
        self.image_data = {
            k: v
            for k, v in self.image_data.items()
            if v["exists"]
        }

        print(
            f"Dataset successfully loaded: "
            f"{len(self.image_data)} unique valid images."
        )

        print(
            f"Total caption rows: {self.total_caption_rows}"
        )

        print(
            f"Empty caption rows: {self.empty_caption_count}"
        )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def validate_dataset(self):
        """
        Perform complete Week 1 dataset integrity validation.
        """

        print("\n" + "=" * 60)
        print("FLICKR30K DATASET VALIDATION")
        print("=" * 60)

        total_images = len(self.image_data)

        total_valid_captions = sum(
            len(data["captions"])
            for data in self.image_data.values()
        )

        caption_counts = [
            len(data["captions"])
            for data in self.image_data.values()
        ]

        print("\nDataset Statistics")
        print("-" * 60)

        print(f"Total valid images: {total_images}")

        print(
            f"Total raw caption rows: "
            f"{self.total_caption_rows}"
        )

        print(
            f"Total valid captions: "
            f"{total_valid_captions}"
        )

        if caption_counts:
            print(
                f"Minimum captions/image: "
                f"{min(caption_counts)}"
            )

            print(
                f"Maximum captions/image: "
                f"{max(caption_counts)}"
            )

        exactly_five = sum(
            count == self.EXPECTED_CAPTIONS_PER_IMAGE
            for count in caption_counts
        )

        print(
            f"Images with exactly 5 captions: "
            f"{exactly_five}"
        )

        # ------------------------------------------------------
        # Missing images
        # ------------------------------------------------------

        missing_images = [
            image_id
            for image_id, data in self.image_data.items()
            if not Path(data["image_path"]).exists()
        ]

        print("\nImage File Validation")
        print("-" * 60)

        print(
            f"Missing image files: "
            f"{len(missing_images)}"
        )

        # ------------------------------------------------------
        # Empty captions
        # ------------------------------------------------------

        print("\nCaption Validation")
        print("-" * 60)

        print(
            f"Empty caption rows: "
            f"{self.empty_caption_count}"
        )

        if self.empty_caption_records:
            for record in self.empty_caption_records:
                print(
                    f"  Row {record['row']}: "
                    f"{record['image_id']}"
                )

        empty_caption_images = [
            image_id
            for image_id, data in self.image_data.items()
            if len(data["captions"]) == 0
        ]

        print(
            f"Images with no valid captions: "
            f"{len(empty_caption_images)}"
        )

        # ------------------------------------------------------
        # Corrupted image check
        # ------------------------------------------------------

        corrupted_images = []

        print("\nChecking image integrity...")

        for image_id, data in self.image_data.items():

            image_path = Path(data["image_path"])

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception:
                corrupted_images.append(image_id)

        print(
            f"Corrupted/unreadable images: "
            f"{len(corrupted_images)}"
        )

        # ------------------------------------------------------
        # Sample mapping
        # ------------------------------------------------------

        print("\n" + "=" * 60)
        print("SAMPLE IMAGE-CAPTION MAPPING")
        print("=" * 60)

        if self.image_data:

            sample_id = next(iter(self.image_data))
            sample = self.image_data[sample_id]

            print(f"\nImage: {sample_id}")
            print(f"Path: {sample['image_path']}")

            for i, caption in enumerate(
                sample["captions"],
                start=1,
            ):
                print(
                    f"Caption {i}: {caption}"
                )

        # ------------------------------------------------------
        # Determine validation status
        # ------------------------------------------------------

        image_count_pass = (
            total_images == self.EXPECTED_IMAGES
        )

        raw_caption_count_pass = (
            self.total_caption_rows
            == self.EXPECTED_CAPTIONS
        )

        missing_images_pass = (
            len(missing_images) == 0
        )

        corrupted_images_pass = (
            len(corrupted_images) == 0
        )

        caption_quality_warning = (
            self.empty_caption_count > 0
            or exactly_five < self.EXPECTED_IMAGES
        )

        if (
            image_count_pass
            and raw_caption_count_pass
            and missing_images_pass
            and corrupted_images_pass
            and not caption_quality_warning
        ):
            status = "PASS"

        elif (
            image_count_pass
            and raw_caption_count_pass
            and missing_images_pass
            and corrupted_images_pass
        ):
            status = "PASS WITH WARNING"

        else:
            status = "FAIL"

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\nStatus: {status}")

        if self.empty_caption_count > 0:
            print(
                "\nDataset-quality warning:"
            )
            print(
                "One or more caption rows are empty."
            )

        print(
            "\nOriginal dataset has NOT been modified."
        )

        return {
            "status": status,
            "total_images": total_images,
            "total_caption_rows": self.total_caption_rows,
            "total_valid_captions": total_valid_captions,
            "exactly_five_captions": exactly_five,
            "empty_caption_rows": self.empty_caption_count,
            "empty_caption_records": self.empty_caption_records,
            "missing_images": len(missing_images),
            "empty_caption_images": len(empty_caption_images),
            "corrupted_images": len(corrupted_images),
        }

    # ==========================================================
    # DATASET SPLITS
    # ==========================================================

    def setup_splits(self, seed: int = 42):
        """
        Load or generate deterministic Train/Validation/Test splits.
        """

        if SPLIT_INFO_PATH.exists():

            with open(
                SPLIT_INFO_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                self.splits = json.load(f)

            # Keep only currently valid images
            for split_name in self.splits:

                self.splits[split_name] = [
                    img_id
                    for img_id in self.splits[split_name]
                    if img_id in self.image_data
                ]

            print(
                f"Loaded existing dataset splits from "
                f"'{SPLIT_INFO_PATH.name}'."
            )

        else:

            all_ids = sorted(
                list(self.image_data.keys())
            )

            rng = random.Random(seed)
            rng.shuffle(all_ids)

            test_ids = all_ids[
                :TEST_SPLIT_SIZE
            ]

            val_ids = all_ids[
                TEST_SPLIT_SIZE:
                TEST_SPLIT_SIZE + VAL_SPLIT_SIZE
            ]

            train_ids = all_ids[
                TEST_SPLIT_SIZE + VAL_SPLIT_SIZE:
            ]

            self.splits = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "full": all_ids,
            }

            with open(
                SPLIT_INFO_PATH,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.splits,
                    f,
                    indent=2,
                )

            print(
                f"Generated new dataset splits: "
                f"Train={len(train_ids)}, "
                f"Val={len(val_ids)}, "
                f"Test={len(test_ids)}."
            )

    # ==========================================================
    # PUBLIC HELPERS
    # ==========================================================

    def get_image_ids(
        self,
        split: str = "full",
    ) -> List[str]:

        if split not in self.splits:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available: {list(self.splits.keys())}"
            )

        return self.splits[split]

    def get_metadata_for_split(
        self,
        split: str = "full",
    ) -> List[Dict]:

        image_ids = self.get_image_ids(split)

        return [
            {
                "index": i,
                "image_id": img_id,
                "image_path": self.image_data[img_id][
                    "image_path"
                ],
                "captions": self.image_data[img_id][
                    "captions"
                ],
            }
            for i, img_id in enumerate(image_ids)
        ]