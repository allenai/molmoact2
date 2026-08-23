import datasets
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("dataset_name", "class_name"),
    [
        ("pixmo_points_train", "PixMoPoints"),
        ("cosyn_point", "CoSynPoint"),
        ("pixmo_multi_points", "PixMoMultiPoints"),
        ("pixmo_multi_image_qa", "PixMoMultiImageQa"),
        ("pixmo_multi_image_qa_multi_only_max5", "PixMoMultiImageQa"),
    ],
)
def test_public_dataset_aliases_resolve_without_initializing(
    monkeypatch, dataset_name, class_name
):
    monkeypatch.setenv("MOLMO_DATA_DIR", "/unused/molmo-data")
    from olmo.data import get_dataset

    dataset_class = getattr(get_dataset, class_name)

    def fail_if_initialized(*args, **kwargs):
        raise AssertionError("dataset was initialized before download")

    monkeypatch.setattr(dataset_class, "__init__", fail_if_initialized)

    assert get_dataset.get_dataset_class_by_name(dataset_name) is dataset_class


def test_download_dataset_by_name_calls_download_before_initialization(monkeypatch):
    monkeypatch.setenv("MOLMO_DATA_DIR", "/unused/molmo-data")
    from olmo.data import get_dataset

    calls = []

    def fail_if_initialized(*args, **kwargs):
        raise AssertionError("dataset was initialized before download")

    def record_download(cls, n_procs):
        calls.append((cls, n_procs))

    monkeypatch.setattr(get_dataset.PixMoPoints, "__init__", fail_if_initialized)
    monkeypatch.setattr(
        get_dataset.PixMoPoints, "download", classmethod(record_download)
    )

    get_dataset.download_dataset_by_name("pixmo_points_train", n_procs=3)

    assert calls == [(get_dataset.PixMoPoints, 3)]


def test_dataset_initializers_are_restored_after_failed_resolution(monkeypatch):
    monkeypatch.setenv("MOLMO_DATA_DIR", "/unused/molmo-data")
    from olmo.data import get_dataset

    original_initializer = get_dataset.PixMoPoints.__init__

    with pytest.raises(NotImplementedError):
        get_dataset.get_dataset_class_by_name("not_a_real_dataset")

    assert get_dataset.PixMoPoints.__init__ is original_initializer


def test_public_image_download_entries_resolve_to_downloadable_classes(monkeypatch):
    monkeypatch.setenv("MOLMO_DATA_DIR", "/unused/molmo-data")
    from olmo.data import get_dataset
    from scripts.download_datasets import DATASET_GROUPS, _flatten_dataset_names

    dataset_names = {
        name
        for group_name in ["pixmo", "image_pointing"]
        for name in _flatten_dataset_names(DATASET_GROUPS[group_name])
    }
    dataset_names.update(
        name
        for name in _flatten_dataset_names(DATASET_GROUPS["demo"])
        if name.startswith("pixmo")
    )

    for name in dataset_names:
        dataset_class = get_dataset.get_dataset_class_by_name(name)
        assert issubclass(dataset_class, get_dataset.Dataset), name
        assert dataset_class.download.__func__ is not get_dataset.Dataset.download.__func__, name


def test_nested_image_urls_are_downloaded_once(monkeypatch, tmp_path):
    from olmo.data import download_urls

    dataset = datasets.Dataset.from_dict(
        {"image_urls": [["https://one", "https://two"], ["https://two", "https://three"]]}
    )
    downloaded = []

    def fake_download(args):
        url = args[0]
        downloaded.append(url)
        return url, str(tmp_path / url.rsplit("/", 1)[-1])

    monkeypatch.setattr(download_urls, "PIXMO_IMAGES", str(tmp_path))
    monkeypatch.setattr(download_urls, "setup_pil", lambda: None)
    monkeypatch.setattr(download_urls, "_download_images", fake_download)

    result = download_urls.download_pixmo_urls(
        dataset,
        n_processes=1,
        check_sha=False,
    )

    assert set(downloaded) == {"https://one", "https://two", "https://three"}
    assert set(result) == set(downloaded)


def test_nested_image_urls_reject_unsupported_sha_validation():
    from olmo.data.download_urls import download_pixmo_urls

    dataset = datasets.Dataset.from_dict({"image_urls": [["https://one"]]})

    with pytest.raises(ValueError, match="nested image_urls"):
        download_pixmo_urls(dataset, n_processes=1, check_sha=True)


def test_cosyn_point_uses_names_published_in_hugging_face_dataset():
    from olmo.data.pixmo_datasets import CoSynPoint

    dataset = object.__new__(CoSynPoint)
    dataset.dataset = [
        {
            "id": "example-id",
            "image": "image-object",
            "questions": ["Where is the cat?"],
            "answer_points": [{"x": [10.0], "y": [20.0]}],
            "names": ["cat"],
        }
    ]

    example = dataset.get(0, np.random.RandomState(0))

    assert example["message_list"][0]["label"] == "cat"
    np.testing.assert_array_equal(
        example["message_list"][0]["points"], np.asarray([[10.0, 20.0]])
    )


def test_pixmo_cap_download_does_not_require_internal_url_map(monkeypatch, tmp_path):
    from olmo.data import pixmo_datasets

    source = datasets.Dataset.from_dict(
        {
            "image_url": ["https://image"],
            "caption": ["caption"],
            "transcripts": [["transcript"]],
        }
    )
    saved = []

    monkeypatch.setattr(pixmo_datasets, "PIXMO_DATASETS", str(tmp_path))
    monkeypatch.setattr(
        pixmo_datasets.datasets, "load_dataset", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(
        pixmo_datasets,
        "download_pixmo_urls",
        lambda *args, **kwargs: {"https://image": "/images/image"},
    )
    monkeypatch.setattr(
        pixmo_datasets,
        "save_local_dataset",
        lambda dataset, name, n_procs, n_val=None: saved.append((dataset, name, n_val)),
    )

    pixmo_datasets.PixMoCap.download(n_procs=1)

    assert len(saved) == 1
    assert saved[0][0]["image"] == ["/images/image"]


def test_pixmo_multi_points_uses_public_metadata_dataset(monkeypatch, tmp_path):
    from olmo.data import pixmo_datasets

    source = datasets.DatasetDict(
        train=datasets.Dataset.from_dict(
            {
                "images": [["/weka/internal/one", "/weka/internal/two"]],
                "image_urls": [["https://one", "https://two"]],
                "labels": [["one", "two"]],
            }
        )
    )
    loaded = []
    saved = []
    image_dir = tmp_path / "images"

    monkeypatch.setattr(
        pixmo_datasets.PixMoMultiPoints, "HOME", str(tmp_path / "multi-points")
    )
    monkeypatch.setattr(pixmo_datasets, "PIXMO_IMAGES", str(image_dir))
    monkeypatch.setattr(
        pixmo_datasets.PixMoPoints,
        "download",
        classmethod(lambda cls, n_procs: None),
    )
    monkeypatch.setattr(pixmo_datasets, "file_exists", lambda path: True)

    def load_dataset(name):
        loaded.append(name)
        return source

    monkeypatch.setattr(pixmo_datasets.datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(
        pixmo_datasets,
        "save_local_dataset",
        lambda dataset, name, n_procs: saved.append((dataset, name)),
    )

    pixmo_datasets.PixMoMultiPoints.download(n_procs=1)

    assert loaded == ["allenai/molmo2-pixmo-multi-points"]
    assert len(saved[0][0]["train"]) == 1
    assert saved[0][0]["train"][0]["images"] == [
        str(image_dir / pixmo_datasets.compute_hash("https://one")),
        str(image_dir / pixmo_datasets.compute_hash("https://two")),
    ]


def test_pixmo_multi_image_qa_uses_public_dataset(monkeypatch, tmp_path):
    from olmo.data import pixmo_datasets

    source = datasets.Dataset.from_dict(
        {
            "image_urls": [["https://one", "https://two"]],
            "image_sha256s": [["sha-one", "sha-two"]],
            "qa_pairs": [
                {"question": ["Question?"], "answer": ["Answer."]}
            ],
        }
    )
    loaded = []
    saved = []
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    monkeypatch.setattr(
        pixmo_datasets.PixMoMultiImageQa, "HOME", str(tmp_path / "multi-image-qa")
    )
    monkeypatch.setattr(pixmo_datasets, "PIXMO_IMAGES", str(image_dir))

    def load_dataset(name, split):
        loaded.append((name, split))
        return source

    monkeypatch.setattr(pixmo_datasets.datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(
        pixmo_datasets,
        "download_pixmo_urls",
        lambda *args, **kwargs: {
            "https://one": str(image_dir / "one"),
            "https://two": str(image_dir / "two"),
        },
    )
    monkeypatch.setattr(
        pixmo_datasets,
        "save_local_dataset",
        lambda dataset, name, n_procs: saved.append((dataset, name)),
    )

    pixmo_datasets.PixMoMultiImageQa.download(n_procs=1)

    assert loaded == [
        ("allenai/Molmo2-MultiImageQA", "train"),
        ("allenai/Molmo2-MultiImageQA", "validation"),
    ]
    assert saved[0][0]["train"][0]["image"] == ["one", "two"]
