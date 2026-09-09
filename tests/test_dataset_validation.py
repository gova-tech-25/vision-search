from src.dataset import Flickr30kDataset


def test_dataset_image_count():
    dataset = Flickr30kDataset()

    assert len(dataset.image_data) == 31783


def test_raw_caption_count():
    dataset = Flickr30kDataset()

    assert dataset.total_caption_rows == 158915


def test_valid_caption_count():
    dataset = Flickr30kDataset()

    valid_captions = sum(
        len(data["captions"])
        for data in dataset.image_data.values()
    )

    assert valid_captions == 158914


def test_empty_caption_detection():
    dataset = Flickr30kDataset()

    assert dataset.empty_caption_count == 1

    assert dataset.empty_caption_records[0]["image_id"] == (
        "2199200615.jpg"
    )


def test_missing_images():
    dataset = Flickr30kDataset()

    missing_images = [
        image_id
        for image_id, data in dataset.image_data.items()
        if not data["exists"]
    ]

    assert len(missing_images) == 0


def test_all_images_have_at_least_one_caption():
    dataset = Flickr30kDataset()

    images_without_captions = [
        image_id
        for image_id, data in dataset.image_data.items()
        if len(data["captions"]) == 0
    ]

    assert len(images_without_captions) == 0


def test_caption_distribution():
    dataset = Flickr30kDataset()

    caption_counts = [
        len(data["captions"])
        for data in dataset.image_data.values()
    ]

    assert min(caption_counts) == 4
    assert max(caption_counts) == 5

    assert caption_counts.count(5) == 31782
    assert caption_counts.count(4) == 1


def test_dataset_validation_status():
    dataset = Flickr30kDataset()

    result = dataset.validate_dataset()

    assert result["status"] == "PASS WITH WARNING"

    assert result["missing_images"] == 0
    assert result["corrupted_images"] == 0
    assert result["empty_caption_rows"] == 1


if __name__ == "__main__":
    dataset = Flickr30kDataset()
    result = dataset.validate_dataset()
    print("\nValidation result:")
    print(result)