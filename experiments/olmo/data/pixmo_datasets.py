import json
import logging
import os
import random
import re
import shutil
from collections import defaultdict
from multiprocessing import Pool
from os.path import basename, exists, join, relpath
from typing import Iterable, List, Literal

import PIL
import datasets
import numpy as np
import torchvision
from cached_path import cached_path
from torchvision.transforms import functional as VF
from PIL import ImageOps, Image
from torchvision.transforms.functional import affine, InterpolationMode
from tqdm import tqdm

from olmo.data.dataset import DATA_HOME, Dataset, DatasetBase, HfDataset, WEB_DATA_HOME
from olmo.data.download_urls import (
    PIXMO_IMAGES,
    compute_hash,
    download_pixmo_urls,
    filter_and_group_data,
)
from olmo.preprocessing.detect_counting_question import is_pixmo_point_and_count_question
from olmo.preprocessing.image_preprocessor import load_pil_image, save_images, load_image
from olmo.util import transpose_dict_of_lists, flatten_lists, resource_path
from olmo.io import read_json, file_exists, is_dir, list_directory, write_file

if DATA_HOME is not None:
    PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
    COSYN_IMAGES = join(DATA_HOME, "cosyn_images")
else:
    PIXMO_DATASETS = None
    COSYN_IMAGES = None
"""Where to save local version of the data after URLs filtering"""


VERIFY = True
"""Verify SSL certificates when downloading"""

NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
"""No-pointing requests templates, used for preprocessing"""


def save_local_dataset(dataset: datasets.Dataset, name: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info("Preparing local dataset...")
    if exists(name):
        logging.info(f"{name} already exists, it will be removed")
        shutil.rmtree(name)
    dataset.save_to_disk(name, num_proc=n_procs)
    logging.info("Done")


class PixMoCount(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "count")
        if exists(local_name):
            return
        all_data = datasets.DatasetDict()
        for split in ["validation", "test", "train"]:
            ds = datasets.load_dataset("allenai/pixmo-count", split=split)
            url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=False)
            ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
            ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, sample=None, counting=False, keep_in_memory=False):
        self.dataset = datasets.load_from_disk(join(PIXMO_DATASETS, "count"), keep_in_memory=keep_in_memory)[split]
        self.counting = counting
        self.split = split

    def __len__(self):
        if self.counting == "both":
            return len(self.dataset) * 2
        else:
            return len(self.dataset)

    def get(self, item, rng):
        if self.counting == "both":
            mode = "point_count" if (item%2==0) else "pointing"
            item = item // 2
        else:
            mode = "point_count" if self.counting else "pointing"

        example = self.dataset[item]
        out = dict(
            style=mode,
            image=example["image"],
            label=example["label"],
            metadata=dict(
                image_url=example["image_url"],
                count=example["count"],
            )
        )
        if self.split == "train":
            points = example["points"]
            out["points"] = np.stack([points["x"], points["y"]], -1, dtype=np.float32)
        return out


def _count_images(folder):
    images = [f for f in list_directory(folder) if f.endswith(".png")]
    try:
        [load_image(img) for img in images]
        return folder, len(images)
    except PIL.UnidentifiedImageError as e:
        return folder, None


class CoSynMultiDocs(Dataset):
    SRC = join(PIXMO_DATASETS, f"pixmo_docs_multi")

    @classmethod
    def download(cls, n_procs=1):
        # pre-compute number of images so we can filter efficiently
        with Pool(n_procs) as pool:
            for doc_type_dir in list_directory(cls.SRC):
                if not is_dir(doc_type_dir):
                    continue
                image_group_metadata = {}
                doc_type = basename(doc_type_dir)
                logging.info(f"Starting {doc_type}")
                metadata_file = join(cls.SRC, f"{doc_type}_metadata_v3.json")
                if file_exists(metadata_file):
                    continue
                folders = list(list_directory(doc_type_dir))
                for folder, image_count in tqdm(pool.imap_unordered(_count_images,folders ), total=len(folders), smoothing=0.7):
                    image_group_metadata[basename(folder)] = image_count
                write_file(cls.SRC, f"{doc_type}_metadata_v3.json", json.dumps(image_group_metadata, indent=2),
                           True)

    def __init__(self, doc_type, use_exp=True, split="train", max_images=None):
        assert split in ["train", "validation"]
        self.doc_type = doc_type
        self.use_exp = use_exp
        with open(resource_path(join(self.SRC, f"{doc_type}_metadata_v3.json"))) as f:
            data = json.load(f)
        n_is_none = sum(x is None for x in data.values())
        data = {k: v for k, v  in data.items() if v is not None and v > 0}
        if max_images is not None:
            filtered = {k: v for k, v  in data.items() if v <= max_images}
            logging.info(f"Filtered down to {len(filtered)} {doc_type} of {len(data)} docs with images<={max_images}, n_invalid={n_is_none}")
            data = filtered
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        src = join(self.SRC, self.doc_type, self.data[idx])
        files = list_directory(src)
        images = [f for f in files if f.endswith(".png")]
        assert len(images) > 0
        images.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))
        with open(join(src, "qa.json")) as f:
            qas = json.load(f)["raw"]
        style = f"cosyn_{self.doc_type}"
        if self.use_exp:
            style += "_exp"
            message_list = [
                dict(question=q["question"], answer=q["answer"], explanation=q["reasoning"], style=style)
                for q in qas
            ]
        else:
            message_list = [
                dict(question=q["question"], answer=q["answer"], style=style)
                for q in qas
            ]
        return dict(
            image=images,
            message_list=message_list,
            metadata=dict(image_paths=images)
        )


class PixMoDocs(Dataset):

    @staticmethod
    def save_image(images: Iterable):
        raise NotImplementedError()
        keys = []
        for image in images:
            key = compute_hash(image["bytes"])
            keys.append(key)
            with open(join(DATA_HOME, "pixmo_docs_images", key), "wb") as f:
                f.write(image["bytes"])
        return dict(image_path=keys)

    @classmethod
    def download(cls, n_procs=1):
        for name in ["other", "charts", "diagrams", "tables"]:
            local_name = join(PIXMO_DATASETS, f"pixmo_docs_{name}")
            if exists(local_name):
                continue
            datasets.load_dataset_builder("allenai/pixmo-docs", name=name).download_and_prepare()
            all_data = datasets.DatasetDict()
            for split in ["validation", "train"]:
                ds = datasets.load_dataset("allenai/pixmo-docs", split=split, name=name)
                ds = ds.cast_column("image", datasets.Image(decode=False))
                # Doing this inplace causes issue with the column feature type,
                # so just map to a new column and then replace the old one
                ds = ds.map(
                    cls.save_image,
                    input_columns="image",
                    batched=True,
                    batch_size=256,
                    num_proc=n_procs if len(ds) > 10000 else 1,
                    desc=f"{name}-{split}-images",
                    remove_columns="image",
                    load_from_cache_file=False
                )
                ds = ds.rename_column("image_path", "image")
                all_data[split] = ds
            save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, doc_type, split, sample=None, keep_in_memory=False, flat=False, use_image_files=True):
        assert doc_type in ["other", "charts", "diagrams", "tables"]
        assert split in ["train", "validation"]
        self.doc_type = doc_type
        self.flat = flat
        self.use_image_files = use_image_files
        if use_image_files:
            # Load a local version of the data that contains filenames instead of the images directly
            local_name = join(PIXMO_DATASETS, f"pixmo_docs_{doc_type}")
            self.dataset = datasets.load_from_disk(local_name, keep_in_memory=keep_in_memory)[split]
        else:
            self.dataset = datasets.load_dataset(
                "allenai/pixmo-docs", name=doc_type, split=split, keep_in_memory=keep_in_memory)
        if flat:
            # Use an index so we don't have to load the images into memory if `keep_in_memory=False`
            # FIXME just switch to the JSON dataset
            logging.info("Building flat index")
            offset = 0
            n_questions = [len(x["question"]) for x in self.dataset["questions"]]
            image_index = np.repeat(np.arange(len(self.dataset), dtype=np.int32), n_questions)
            question_index = np.concatenate([np.arange(x, dtype=np.int32) for x in n_questions], 0)
            self.flat_index = np.stack([image_index, question_index], 1)
            logging.info("Done")

    def __len__(self):
        return len(self.flat_index) if self.flat else len(self.dataset)

    def get(self, item, rng):
        style = f"pixmo_docs_{self.doc_type}"
        if self.flat:
            image_ix, question_ix = self.flat_index[item]
            example = self.dataset[int(image_ix)]
            if self.use_image_files:
                example["image"] = join(DATA_HOME, "pixmo_docs_images", example["image"])
            qas = example["questions"]
            return dict(
                image=example["image"],
                question=qas["question"][question_ix],
                answer=qas["answer"][question_ix],
                style=style,
                metadata=dict(
                    image_id=example["image_id"]
                )
            )
        example = self.dataset[item]
        qas = example["questions"]
        if self.use_image_files:
            example["image"] = join(DATA_HOME, "pixmo_docs_images", example["image"])
        return dict(
            image=example["image"],
            message_list=[
                dict(question=q, answer=a, style=style) for q, a in
                zip(qas["question"], qas["answer"])
            ],
            metadata=dict(
                image_id=example["image_id"]
            )
        )


class PixMoPoints(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False, hold_out_pointing_eval=True):
        collection_method = ["pointing", "counting"]
        local_names = [join(PIXMO_DATASETS, f"points-{name}") for name in collection_method]
        if all(exists(x) for x in local_names):
            return
        ds = datasets.load_dataset("allenai/pixmo-points", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        if hold_out_pointing_eval:
            eval_ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
            for url in eval_ds["image_url"]:
                if url in filenames:
                    del filenames[url]
        for method, local_name in zip(collection_method, local_names):
            logging.info(f"Building subset {method}")
            ds_for_method = ds.filter(lambda x: x == method, input_columns="collection_method")
            filtered_dataset = filter_and_group_data(ds_for_method, filenames, check_sha)
            name = "high_frequency" if method == "counting" else "basic"
            save_local_dataset(filtered_dataset, local_name, n_procs=n_procs, n_val=n_val)

    def __init__(self, split, kind="both", counting=False, keep_in_memory=False,
                 max_points=None, max_total_points_per_example=None):
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.counting = counting
        if counting == "both":
            self.mode = ["point_count", "pointing"]
        else:
            self.mode = "point_count" if counting else "pointing"
        self.split = split
        self.kind = kind
        if kind == "both":
            data1 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-counting"), keep_in_memory=keep_in_memory)[split]
            data2 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-pointing"), keep_in_memory=keep_in_memory)[split]
            self.data = datasets.concatenate_datasets([data1, data2])
        elif kind == "basic":
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-pointing"), keep_in_memory=keep_in_memory)[split]
        else:
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-counting"), keep_in_memory=keep_in_memory)[split]
        if max_total_points_per_example or max_points:
            n_points = self.data["count"][:]
            sub_index = []
            total_points = 0
            n_filtered = 0
            for image_idx, point_counts in enumerate(n_points):
                sub_batches = []
                on = []
                total_on = 0
                total_points += len(point_counts)
                for ix, n in enumerate(point_counts):
                    if max_points and n > max_points:
                        n_filtered += 1
                        continue
                    if max_total_points_per_example and (total_on + n > max_total_points_per_example):
                        if on:
                            sub_batches.append(on)
                            total_on = 0
                            on = []
                    on.append(ix)
                    total_on += n
                if on:
                    sub_batches.append(on)
                for ix in sub_batches:
                    sub_index.append((image_idx, ix))
            logging.info(f"Filtered {n_filtered} ({n_filtered}/{total_points}) points")
            logging.info(f"Split {len(self.data)} examples into {len(sub_index)} parts")
            self.sub_index = sub_index
        else:
            self.sub_index = None

    def __len__(self):
        n = len(self.sub_index) if self.sub_index else len(self.data)
        if self.counting == "both":
            n *= 2
        return n

    def get(self, item, rng):
        if self.counting == "both":
            mode = self.mode[item % 2]
            item = item // 2
        else:
            mode = self.mode

        if self.sub_index:
            image_idx, point_idx = self.sub_index[item]
            ex = dict(self.data[image_idx])
            ex["label"] = [ex["label"][i] for i in point_idx]
            ex["points"] = [ex["points"][i] for i in point_idx]
        else:
            ex = self.data[item]

        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            messages.append(dict(
                label=label,
                points=np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1),
                point_scale=100,
                clip_points=True,
                style=mode
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            )
        )


class PixMoPointExplanations(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "point-explanations")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-point-explanations", split="train")
        ds = ds.filter(lambda x: x is not None, input_columns=["parsed_response"])
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, split_groups=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.split_groups = split_groups
        data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "point-explanations"),
            keep_in_memory=keep_in_memory)[split]
        out = []
        for ex in data:
            molmo_ex = dict(
                image=ex["image"],
                metadata=dict(
                    image_url=ex["image_url"],
                )
            )
            msg_list = []
            for q, res, alt, inline, points in zip(
                ex["question"], ex["parsed_response"],
                ex["alt_text"], ex["inline_text"], ex["points"]
            ):
                msg_list.append(dict(
                    question=q,
                    answer=res,
                    answer_annotations=[dict(
                        points=p, inline_text=i, alt_text=a
                    ) for p, i, a in zip(points, inline, alt)],
                    style="point_qa"
                ))
            if self.split_groups and len(msg_list) > 1:
                n = len(msg_list) // 2 + len(msg_list) % 2
                out.append(dict(molmo_ex, message_list=msg_list[:n]))
                out.append(dict(molmo_ex, message_list=msg_list[n:]))
            else:
                out.append(dict(molmo_ex, message_list=msg_list))
        self.data = out

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return dict(self.data[item])

class PixMoCapQa(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "cap-qa")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap-qa", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False, style="synthetic_qa"):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap-qa"), keep_in_memory=keep_in_memory)[split]
        self.style = style

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        question = example["question"]
        answer = example["answer"]
        message_lists = []
        for qs, ans in zip(question, answer):
            parts = re.split(r"\s*(\[USER\]|\[ASSISTANT\])\s*", qs)
            assert parts[0] == ""
            assert parts[-1] == ""
            parts = parts[1:-1]
            assert len(parts) % 4 == 3
            messages = []
            for part_ix, part in enumerate(parts):
                if part_ix % 4 == 0:
                    assert part == "[USER]"
                elif part_ix % 4 == 1:
                    assert part
                    messages.append(part)
                elif part_ix % 4 == 2:
                    assert part == "[ASSISTANT]"
                else:
                    assert part
                    messages.append(part)
            messages.append(ans)
            message_lists.append(dict(messages=messages, style=self.style))

        example = dict(
            image=example["image"],
            message_list=message_lists,
            metadata=dict(
                image_url=example["image_url"],
            )
        )
        if self.prefix_how_many:
            for conv in example["message_list"]:
                messages = conv["messages"]
                for user_question_ix in range(0, len(messages), 2):
                    question = messages[user_question_ix]
                    answer = messages[user_question_ix+1]
                    if is_pixmo_point_and_count_question(question, answer):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        messages[user_question_ix] = prefix + messages[user_question_ix]
        return example


class PixMoCap(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False, sample=None):
        local_name = join(PIXMO_DATASETS, "cap")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap", split="train")
        if sample:
            ds = ds.take(sample)
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        logging.info("Preparing data...")
        filtered_dataset = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        filtered_dataset = filtered_dataset.add_column(
            "image", [url_to_filename[x] for x in filtered_dataset["image_url"]])
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, mode, prefix_how_many=True, keep_in_memory=False, flatten=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        if mode not in ["transcript", "transcripts", "captions", "transcript_and_caption", "transcript1_and_caption"]:
            raise ValueError(mode)
        self.split = split
        self.mode = mode
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = []
        caption = ex.pop("caption")
        transcripts = ex.pop("transcripts")
        if self.mode in ["captions", "transcript_and_caption", "transcript1_and_caption"]:
            messages.append(dict(text=caption, style="long_caption"))
        if self.mode in ["transcript_and_caption", "transcript1_and_caption", "transcript"]:
            if self.mode == "transcript_and_caption":
                ix = rng.randint(0, len(transcripts))
            else:
                ix = 0
            messages.append(dict(text=transcripts[ix], style="transcript"))
        if self.mode == "transcripts":
            messages += [dict(text=tr, style="transcript") for tr in transcripts]
        out = dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_path=ex["image"],
                image_url=ex.pop("image_url"),
            )
        )
        return out


class PixMoAskModelAnything(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "ask-model-anything")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-ask-model-anything", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False, flat=False,
                 skip_counting=False, sample=None):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.skip_counting = skip_counting
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.flat = flat
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "ask-model-anything"), keep_in_memory=keep_in_memory)[split]
        if self.flat:
            all_questions = self.data["question"]
            n_questions = [len(x) for x in all_questions]
            image_index = np.repeat(np.arange(len(self.data), dtype=np.int32), n_questions)
            question_index = np.concatenate([np.arange(x, dtype=np.int32) for x in n_questions], 0)
            self.flat_index = np.stack([image_index, question_index], 1)
            if self.skip_counting:
                is_counting = flatten_lists(
                    [re.fullmatch("how many.*", q.strip(), flags=re.IGNORECASE) is not None for q in questions]
                    for questions in all_questions)
                assert len(is_counting) == len(self.flat_index)
                self.flat_index = self.flat_index[~np.array(is_counting)]
            if sample:
                np.random.RandomState(872).shuffle(self.flat_index)
                self.flat_index = self.flat_index[:sample]
        else:
            if skip_counting or sample:
                raise NotImplementedError()

    def __len__(self):
        return len(self.flat_index) if self.flat else len(self.data)

    def get(self, item, rng):
        if self.flat:
            item, question_ix = self.flat_index[item]
            example = self.data[int(item)]
            q = example["question"][question_ix].strip()
            a = example["answer"][question_ix]
            metadata = dict(question=q, answer=a, image_file=example["image"])
            messages = [dict(question=q, answer=a, style="user_qa")]
        else:
            question_id = None
            example = self.data[item]
            messages = []
            for q, a in zip(example["question"], example["answer"]):
                messages.append(dict(question=q.strip(), answer=a, style="user_qa"))
            metadata = dict(image_url=example["image_url"])

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=metadata
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                if is_pixmo_point_and_count_question(conv["question"], conv["answer"]):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    conv["question"] = prefix + conv["question"]
        return ex


class PixMoPointsEval(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, cache_only=False):
        local_name = join(PIXMO_DATASETS, "pixmo-points-eval")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, keep_in_memory=False, legacy=False):
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "pixmo-points-eval"), keep_in_memory=keep_in_memory)
        self.legacy = legacy

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        points = ex["points"]
        messages = []
        points = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
        mask = np.array(ex["masks"], dtype=bool)

        if self.legacy:
            gt_points = points
        else:
            h, w = mask.shape[1:]
            gt_points = points * np.array([w, h])[None, :]/100

        return dict(
            image=ex["image"],
            label=ex["label"],
            points=points,
            point_scale=100,
            style="pointing",
            metadata=dict(
                label=ex["label"],
                masks=mask,
                image_url=ex["image_url"],
                gt_points=gt_points
            )
        )


class DenseCaptionEval(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __init__(self):
        with open(cached_path(join(PIXMO_DATASETS, "dense-caption-eval", "test.jsonl")), "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def get(self, item, rng):
        ex = json.loads(self.lines[item])
        return dict(
            image=join(DATA_HOME, "pixmo_images", ex["image"]),
            style="long_caption",
            metadata=dict(
                image_url=ex["url"],
            )
        )


class PixMoClocks(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError("Created from the original tfrecords")

    def __init__(self, split, aug=True):
        self.aug = aug
        super().__init__(split)

    def load(self):
        split = self.split
        src = join(PIXMO_DATASETS, "clocks", f"{split}.jsonl")
        logging.info(f"Loading pixmo clock data from {src}")
        with open(cached_path(src, cache_dir=os.environ.get("MOLMO_CACHE_DIR"))) as f:
            return f.readlines()

    def get(self, item, rng: np.random.RandomState):
        ex = json.loads(self.data[item])

        time_format = ex["time_format"]
        shows_seconds = ex["shows_seconds"]
        hour, minute, second = [int(ex[k]) for k in ["hour", "minute", "second"]]
        if hour == 0:
            hour_str = "12"  # Midnight of the previous day
            am_pm = "AM"
        elif hour > 12:
            am_pm = "PM"
            hour_str = hour - 12
        else:
            hour_str = hour
            am_pm = "AM"
        hour_str = str(hour_str)
        minute_str = str(minute)
        if len(minute_str) == 1:
            minute_str = "0" + minute_str
        second_str = str(second)

        if len(second_str) == 1:
            second_str = "0" + second_str

        prefix = "The time shown is "
        if time_format == "The time is not shown":
            text = "The time is not shown in the image."
            hour, minute, second = -1, -1, -1
        else:
            if not shows_seconds:
                second = -1
            if time_format == "12 hour clock (without AM/PM)" and shows_seconds:
                if hour >= 12:
                    hour = hour - 12
                time = "".join([hour_str, ":", minute_str, ":", second_str])
            elif time_format == "12 hour clock (with AM/PM)" and shows_seconds:
                time = "".join([hour_str, ":", minute_str, ":", second_str, " ", am_pm])
            elif time_format == "12 hour clock (with AM/PM)" and not shows_seconds:
                time = "".join([hour_str, ":", minute_str, " ", am_pm])
            elif time_format == "12 hour clock (without AM/PM)" and not shows_seconds:
                if hour >= 12:
                    hour = hour - 12
                time = "".join([hour_str, ":", minute_str])
            else:
                raise RuntimeError()
            text = "".join(["The time shown is ", time])

        image = load_pil_image(join(PIXMO_DATASETS, "clocks", "images", ex["image"]))
        # Cutoff the black sharding at the bottom of every image
        image = image.crop((0, 0, image.width, image.height-120))

        if self.aug:
            sel = rng.random()
            if sel < 0.1:
                # Straight on
                shear_x = 0.
                shear_y = 0.
                rotation = 0.
            elif sel < 0.5:
                # Normal looking
                shear_x = rng.uniform(-10, 10)
                shear_y = rng.uniform(-10, 10)
                rotation = rng.uniform(-25, 25)
            else:
                if rng.random() > 0.5:
                    shear_x = rng.uniform( -30, 30)
                    shear_y = rng.uniform( -30, 30)
                else:
                    shear_x = rng.uniform( -10, 10)
                    shear_y = rng.uniform( -10, 10)
                rot_rng = rng.random()
                if rot_rng < 0.2:
                    rotation = rng.uniform( -25, 25)
                elif rot_rng < 0.6:
                    rotation = rng.uniform( -80, 80)
                else:
                    rotation = rng.uniform( -180, 180)

            if rng.random() > 0.5:
                scale = rng.uniform(0.3, 2)
            else:
                scale = rng.uniform(0.3, 1)

            # Avoid parts of the clock getting cutoff by the affine transform
            image = torchvision.transforms.Pad([200, 200, 200, 200], fill=255)(image)
            shear_y, shear_x = 0, 0
            image = affine(
                image,
                rotation,
                translate=[0, 0],
                scale=scale,
                shear=[shear_x, shear_y],
                interpolation=InterpolationMode.BILINEAR,
                fill=255
            )

            # Crop to whitespace
            bbox = ImageOps.invert(image).getbbox()
            image = image.crop(bbox)

            # Translate so the clock is not in the center
            height, width = image.height, image.width
            if rng.random() < 0.2:
                h_pad = rng.randint(0, height//2, (2,), dtype=np.int32)
                w_pad = rng.randint(0, width//2, (2,), dtype=np.int32)
            else:
                h_pad = rng.randint(0, height*2, (2,), dtype=np.int32)
                w_pad = rng.randint(0, width*2, (2,), dtype=np.int32)
            image = torchvision.transforms.Pad([h_pad[0], w_pad[0], h_pad[1], w_pad[1]], fill=255)(image)

            # Mild color jitter
            image = VF.adjust_hue(image, rng.uniform(-0.05, 0.05))
            image = VF.adjust_brightness(image, rng.uniform(0.85, 1.2))
            image = VF.adjust_saturation(image, rng.uniform(0.8, 1.2))
            image = VF.adjust_contrast(image, rng.uniform(0.8, 1.2))

        return dict(
            image=np.array(image),
            prompt="What time is being shown?",
            text=text,
            metadata=dict(hour=hour, second=second, minute=minute),
            style="clocks"
        )


class CoSyn(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        for name in [
            "chart", "chemical", "circuit", "diagram",
            "document", "graphic", "math", "music",
            "nutrition", "table"
        ]:
            local_name = join(PIXMO_DATASETS, f"cosyn-{name}")
            if exists(local_name):
                continue
            all_data = datasets.DatasetDict()
            for split in ["train", "validation"]:
                ds = datasets.load_dataset("allenai/CoSyn-400K", name=name, split=split)
                pil_images = (ex["image"] for ex in ds)
                filenames = [
                    join(COSYN_IMAGES, name, f"{img_id}.png")
                    for img_id in ds["id"]
                ]
                saved_images = save_images(pil_images, filenames, n_procs)
                assert len(saved_images) == len(filenames)
                def pil_to_path(ex):
                    ex["image"] = join(COSYN_IMAGES, name, f"{ex['id']}.png")
                    return ex
                new_features = ds.features.copy()
                new_features["image"] = datasets.Value("string")
                ds = ds.map(pil_to_path, features=new_features)
                all_data[split] = ds
            save_local_dataset(all_data, local_name, n_procs)
    
    def __init__(self, doc_type, split, use_exp=True, keep_in_memory=False, flat=False):
        assert doc_type in [
            "chart", "chemical", "circuit", "diagram",
            "document", "graphic", "math", "music",
            "nutrition", "table"
        ]
        assert split in ["train", "validation"]
        self.doc_type = doc_type
        self.split = split
        self.use_exp = use_exp
        self.dataset = datasets.load_from_disk(
            join(PIXMO_DATASETS, f"cosyn-{doc_type}"), keep_in_memory=keep_in_memory)[split]
        self.flat = flat
        if flat:
            logging.info("Building flat index")
            offset = 0
            n_questions = [len(x["question"]) for x in self.dataset["qa_pairs"]]
            image_index = np.repeat(np.arange(len(self.dataset), dtype=np.int32), n_questions)
            question_index = np.concatenate([np.arange(x, dtype=np.int32) for x in n_questions], 0)
            self.flat_index = np.stack([image_index, question_index], 1)
            logging.info("Done")

    def __len__(self):
        return len(self.flat_index) if self.flat else len(self.dataset)

    def get(self, item, rng):
        if self.flat:
            item, question_ix = self.flat_index[item]
        else:
            question_ix = None
        style = f"cosyn_{self.doc_type}"
        example = self.dataset[int(item)]
        qeas = example["qa_pairs"]
        if self.use_exp:
            style += "_exp"
            message_list = [
                dict(question=q, explanation=e, answer=a, style=style) for q, e, a in
                zip(qeas["question"], qeas["explanation"], qeas["answer"])
            ]
        else:
            message_list = [
                dict(question=q, answer=a, style=style) for q, a in
                zip(qeas["question"], qeas["answer"])
            ]

        if self.flat:
            message_list = [message_list[question_ix]]
        return dict(
            image=example["image"],
            message_list=message_list,
            metadata=dict(
                image_id=example["id"]
            )
        )


class CoSynPoint(HfDataset):
    PATH = "allenai/CoSyn-point"

    def __init__(self, split, keep_in_memory=False):
        super().__init__(split, keep_in_memory=keep_in_memory)

    def get(self, item, rng):
        example = self.dataset[item]
        messages = []
        for question, points, name in zip(example["questions"], example["answer_points"], example["names"]):
            messages.append(dict(
                question=question,
                points=np.stack([points['x'], points['y']], -1),
                label=name,
                point_scale=100,
                style="cosyn_point",
            ))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_id=example["id"]
            )
        )


class CorrectionQa(Dataset):
    PREFIX = "https://explore-multimodal-datasets.s3.us-west-2.amazonaws.com/correction-urls"
    
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()
    
    def __init__(self, split, multi_image_only=False, max_images=None, prefix_how_many=True):
        assert split in ["train", "validation"]
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.max_images = max_images
        self.multi_image_only = multi_image_only
        self.dataset = self.load()
    
    def load(self):
        split = self.split
        local_data_name = cached_path(
            join(PIXMO_DATASETS, "correction-qa", f"{split}-records.json"),
            cache_dir=os.environ.get("MOLMO_CACHE_DIR"),
        )
        with open(local_data_name, 'r') as f:
            records = json.load(f)
        
        group_by_images = defaultdict(list)
        for record in records:
            if "imageUrls" in record:
                group_by_images[tuple(record["imageUrls"])].append(record)
            else:
                group_by_images[tuple([record["imageUrl"]])].append(record)
        
        if self.multi_image_only:
            group_by_images = {k: v for k, v in group_by_images.items() if len(k) > 1}
        if self.max_images:
            group_by_images = {k: v for k, v in group_by_images.items() if len(k) <= self.max_images}

        out = []
        for image_urls, records in group_by_images.items():
            image = list(image_urls)
            questions = []
            answers = []
            for record in records:
                questions.append(record["question"])
                answers.append(record["answer"])
            out.append(
                dict(
                    image=image,
                    questions=questions,
                    answers=answers,
                    prolificId=[record["prolificId"] for record in records],
                    obejctID=[record["objectID"] for record in records],
                )
            )
        return out
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng: np.random.RandomState):
        example = self.dataset[item]
        image = example["image"]
        dst_prefix = join(DATA_HOME, "correction_images")
        if len(image) > 1:
            image = [url.replace(self.PREFIX, dst_prefix) for url in image]
        else:
            image = image[0].replace(self.PREFIX, dst_prefix)
        messages = []
        for q, a in zip(example["questions"], example["answers"]):
            if self.prefix_how_many:
                if is_pixmo_point_and_count_question(q, a):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    q = prefix + q
            messages.append(dict(question=q, answer=a, style="correction_qa"))
        rng.shuffle(messages)

        return dict(
            image=image,
            message_list=messages,
        )


class PixMoMultiImageQa(Dataset):
    HOME = join(PIXMO_DATASETS, "pixmo-multi-image-qa")

    @classmethod
    def download(cls, n_procs=1, check_sha=False, cache_only=False):
        if exists(cls.HOME):
            return
        dataset = datasets.DatasetDict()
        for split in ["train", "validation"]:
            ds = datasets.load_dataset("allenai/Molmo2-MultiImageQA", split=split)
            url_to_filename = download_pixmo_urls(
                ds,
                n_procs,
                check_sha=check_sha,
                cache_only=cache_only,
                verify=False,
            )
            ds = ds.filter(
                lambda urls: all(url in url_to_filename for url in urls),
                input_columns=["image_urls"],
            )
            ds = ds.add_column(
                "image",
                [
                    [relpath(url_to_filename[url], PIXMO_IMAGES) for url in urls]
                    for urls in ds["image_urls"]
                ],
            )
            dataset[split] = ds
        save_local_dataset(dataset, cls.HOME, n_procs)

    def __init__(
        self,
        split,
        multi_image_only=False,
        max_images=None,
        prefix_how_many=True,
        keep_in_memory=False,
    ):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.multi_image_only = multi_image_only
        self.max_images = max_images
        self.dataset = datasets.load_from_disk(
            self.HOME, keep_in_memory=keep_in_memory
        )[split]
        if max_images is not None:
            minimum_images = 2 if multi_image_only else 1
            self.dataset = self.dataset.filter(
                lambda urls: minimum_images <= len(urls) <= max_images,
                input_columns=["image_urls"],
            )
        elif multi_image_only:
            self.dataset = self.dataset.filter(
                lambda urls: len(urls) >= 2,
                input_columns=["image_urls"],
            )
        else:
            self.dataset = self.dataset.filter(
                lambda urls: len(urls) >= 2,
                input_columns=["image_urls"],
            )

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = self.dataset[item]
        messages = []
        for question, answer in zip(
            example["qa_pairs"]["question"], example["qa_pairs"]["answer"]
        ):
            if self.prefix_how_many and is_pixmo_point_and_count_question(question, answer):
                prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                question = prefix + question
            messages.append(dict(question=question, answer=answer, style="correction_qa"))
        return dict(
            image=[join(PIXMO_IMAGES, path) for path in example["image"]],
            message_list=messages,
            metadata=dict(image_urls=example["image_urls"]),
        )


class PixMoMultiImageCapQa(Dataset):
    FORMATS = ["answer_first", "answer_last", "short_answer"]

    @classmethod
    def download(cls, n_procs=1, n_val=2048):
        local_name = join(PIXMO_DATASETS, "pixmo-multi-image-cap-qa")
        if exists(local_name):
            return
        qas = read_json(join(PIXMO_DATASETS, "pixmo-cap-multi-img-qa-fix-list.json"))
        ds = datasets.Dataset.from_list(qas)
        save_local_dataset(ds, local_name, n_procs, n_val=n_val)
    
    def __init__(self, split, format: str = "all", keep_in_memory=False):
        assert split in ["train", "validation"]
        assert format in ["all"] + self.FORMATS
        self.split = split
        self.dataset = datasets.load_from_disk(
            join(PIXMO_DATASETS, "pixmo-multi-image-cap-qa"), keep_in_memory=keep_in_memory)[split]
        if format != "all":
            ds = self.dataset.filter(lambda x: x["format"] == format)
            self.dataset = ds
    
    def is_candidate_qa(self, question: str, answer: str) -> bool:
        q_matches = re.findall(r'\([A-H]\)|[A-H][\.\)]', question)
        a_matches = re.findall(r'\([A-H]\)|[A-H][\.\)]', answer)
        return len(q_matches) == len(set(q_matches)) and len(q_matches) >= 2 and len(a_matches) == 1 and set(a_matches) <= set(q_matches)

    def format_answer_last_question(self, question: str) -> str:
        parts = question.split("\n", 1)
        if len(parts) < 2:
            question = question.rstrip() + " Explain your reasoning and then choose the correct option."
        else:
            assert len(parts) == 2
            question = "\n".join(
                [
                    parts[0].rstrip() + " Explain your reasoning and then choose the correct option.",
                    parts[1],
                ]
            )
        return question
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng):
        example = self.dataset[item]
        question = example["question"]
        answer = example["answer"]
        format = example["format"]
        if self.is_candidate_qa(question, answer) and rng.random() < 0.1:
            question = re.sub(r'\(([A-H])\)|([A-H])[\.\)]', lambda m: f"{m.group(1) or m.group(2)}:", question)
            answer = re.sub(r'\(([A-H])\)|([A-H])[\.\)]', lambda m: f"{m.group(1) or m.group(2)}:", answer)
        
        if format == "short_answer" and re.match(r'^[A-H][\)\.:]', answer) and rng.random() < 0.1:
            answer = answer[0]
        
        if format == "answer_last":
            question = self.format_answer_last_question(question)
        
        style = f"multi_image_{format}" if format != "short_answer" else "multi_image_mc"

        out = dict(
            image=example["images"],
            question=question,
            answer=answer,
            style=style,
            metadata=dict(
                format=format,
                category=example["category"],
                image_urls=example["image_urls"],
            )
        )

        return out


class PixMoMultiImageMMRCapQa(Dataset):
    """
    Group pixmo images using the MMR algorithm.
    Prompt an LLM to generate QAs for each group of images.
    2-5 images per question
    """

    @classmethod
    def download(cls, n_procs=1, n_val=2048):
        local_name = join(PIXMO_DATASETS, "pixmo-mmr-multi-image-cap-qa_sent_reasoning_1m")
        if file_exists(local_name):
            return
        qas = read_json(join(PIXMO_DATASETS, "pixmo-mmr-multi-image-cap-qa_sent_reasoning_filtered_1m_cleaned.json"))
        ds = datasets.Dataset.from_list(list(qas.values()))
        save_local_dataset(ds, local_name, n_procs, n_val=n_val)
    
    def __init__(self, split, keep_in_memory=False, enable_cot=False):
        assert split in ["train", "validation"]
        self.split = split
        self.enable_cot = enable_cot
        
        # Check if dataset exists, download if not
        local_name = join(PIXMO_DATASETS, "pixmo-mmr-multi-image-cap-qa_sent_reasoning_1m")
        if not file_exists(local_name):
            print(f"Dataset not found at {local_name}. Attempting to download...")
            self.download()
        
        self.dataset = datasets.load_from_disk(local_name, keep_in_memory=keep_in_memory)[split]

    def format_answer_last_question(self, question: str, options: List[str], sep: str = ".") -> str:
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}{sep} {options[idx]}" for idx in range(len(options))
        )
        question = question.rstrip() + " Explain your reasoning and then choose the correct option."
        prompts = [question, option_text]
        prompt = "\n".join(prompts)
        return prompt
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng):
        example = self.dataset[item]

        question = example["question"]
        choices = [item['text'] for item in example["choices"]] # XXX temporary placeholder.
        reasoning = example["reason"]
        answer = example["answer"]
        category = example["category"]

        out = dict(
            image=example["selected_images"],
            metadata=dict(
                category=category,
            )
        )

        if self.enable_cot:
            out.update(
                question=self.format_answer_last_question(question, choices),
                answer=answer,
                style="multi_image_mc_exp",
                explanation=reasoning,
            )
        else:
            out.update(
                question=question,
                options=choices,
                answer_idx=ord(answer) - ord("A"),
                style="multi_image_mc",
            )

        return out


class PixMoMultiPoints(Dataset):
    MULTI_IMAGE_POINTING_STYLES = [
            "multi_image_pointing",
            "multi_image_point_then_count",
        ]
    HOME = join(PIXMO_DATASETS, "pixmo-multi-points")

    @classmethod
    def download(cls, n_procs=1):
        if exists(cls.HOME):
            return
        PixMoPoints.download(n_procs=n_procs)
        dataset = datasets.load_dataset("allenai/molmo2-pixmo-multi-points")
        image_exists_cache = {}

        def use_local_images(example):
            example["images"] = [
                join(PIXMO_IMAGES, compute_hash(url)) for url in example["image_urls"]
            ]
            return example

        dataset = dataset.map(use_local_images, num_proc=n_procs)

        def images_exist(image_paths):
            for image_path in image_paths:
                if image_path not in image_exists_cache:
                    image_exists_cache[image_path] = file_exists(image_path)
                if not image_exists_cache[image_path]:
                    return False
            return True

        dataset = dataset.filter(
            images_exist,
            input_columns=["images"],
            num_proc=n_procs,
        )
        save_local_dataset(dataset, cls.HOME, n_procs)
    
    def __init__(self, split, keep_in_memory=False,
                 styles=("multi_image_pointing", "multi_image_point_then_count")):
        assert split in ["train", "validation"]
        assert all(x in self.MULTI_IMAGE_POINTING_STYLES for x in styles)
        self.styles = styles
        self.split = split
        # Load the dataset
        if not is_dir(self.HOME):
            # Try to download if it does not exist
            PixMoMultiPoints.download()
        self.dataset = datasets.load_from_disk(
            self.HOME, keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = dict(self.dataset[item])
        example["style"] = rng.choice(self.styles)
        example['image'] = example['images']
        example['point_scale'] = 100
        example['clip_points'] = True
        example["metadata"] = dict(image_paths=example['images'])

        return example


def gaussian_sample_around_bbox_center(
    bbox, margin_percent=0.1, max_dist_from_center=None
):
    """
    Generate a random point around the center of a bounding box using Gaussian distribution.
    The standard deviation is set to 1/3 * (1 - margin_percent) of the distance to boundaries.
    The sampled x and y are optionally clipped to be within max_dist_from_center after sampling.
    Args:
        bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        tuple: A randomly sampled (x, y) coordinate.
    """
    x_min, y_min, x_max, y_max = bbox

    # Compute the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2

    x_margin = (
        1 - margin_percent
    ) * half_width  # sample x within (1 - margin_percent) of half_width
    y_margin = (
        1 - margin_percent
    ) * half_height  # sample y within (1 - margin_percent) of half_height

    x_sigma = x_margin / 3
    y_sigma = y_margin / 3

    # Sample from Gaussian distribution centered at the center
    sampled_x = random.gauss(center_x, x_sigma)
    sampled_y = random.gauss(center_y, y_sigma)

    if (
        max_dist_from_center
    ):  # clip x, y based on max_dist_from_center if provided
        sampled_x = min(sampled_x, center_x + max_dist_from_center)
        sampled_x = max(sampled_x, center_x - max_dist_from_center)

        sampled_y = min(sampled_y, center_y + max_dist_from_center)
        sampled_y = max(sampled_y, center_y - max_dist_from_center)

    return (sampled_x, sampled_y)


def get_click_coords_from_bbox(bbox, mode="center"):
    """
    Get the click point from the bounding box.
    Args:
        bbox: a list of four coordinates [x1, x2, y1, y2]
        mode: the mode to get the click point. "center", "top_left", "random_gaussian", or "random_uniform"
    Returns:
        x: the x coordinate of the click point
        y: the y coordinate of the click point
    """
    assert len(bbox) == 4, f"Invalid bbox: {bbox}"

    x1, y1, x2, y2 = bbox
    assert x2 >= x1 and y2 >= y1, f"Invalid bbox values: {bbox}"

    if mode == "top_left":
        return x1, y1
    elif mode == "center":
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return x, y
    elif mode == "random_uniform":
        x = random.uniform(x1, x2)
        y = random.uniform(y1, y2)
        return x, y
    elif mode == "random_gaussian":
        x, y = gaussian_sample_around_bbox_center(bbox)
        return x, y
    else:
        raise ValueError(f"Unknown mode: {mode}")


def normalize_click_coords(
    x, y, image_w, image_h, upper_bound=100, num_digits=1
):
    """
    Normalize the coordinates to [0, upper_bound]
    Args:
        x: the x coordinate
        y: the y coordinate
        image_w: the width of the image
        image_h: the height of the image
        upper_bound: the upper bound of the normalized coordinates
        num_digits: the number of digits to round to
    Returns:
        x: the normalized x coordinate
        y: the normalized y coordinate
    """
    x = round(x / image_w * upper_bound, num_digits)
    y = round(y / image_h * upper_bound, num_digits)
    # add min and max clipping to ensure normalized coords are between 0 and upperbound
    x = max(0, min(x, upper_bound))
    y = max(0, min(y, upper_bound))
    return x, y


class SyntheticGround(DatasetBase):
    """
    Synthetic dataset for web grounding tasks.
    """
    data_path = join(WEB_DATA_HOME, "webolmo_synthetic_ground")
    ALL_WEBSITES = [
        "allrecipes", "amazon", "apple", "arxiv", "bbc_news", "bbc_news_v2",
        "booking", "cambridge_dictionary", "coursera", "espn", "github",
        "google_maps", "google_travel", "gscholar", "gsearch", "huggingface",
        "papers_with_code", "perplexity", "s2", "satlas", "scholar_qa",
        "wikipedia", "wolfram_alpha",
    ]

    def __init__(
        self,
        split: Literal["train", "val", "val_iid", "val_ood"],
        mode="random_gaussian",
        clickable_only: bool = True,
        flatten: bool = False,
        action_only: bool = False,
        max_msg_per_screenshot: int = -1,
        style: str = "web_grounding"
    ):
        self.split = "val" if split == "validation" else split
        self.mode = mode
        self.clickable_only = clickable_only
        self.action_only = action_only
        self.flatten = flatten
        self.max_msg_per_screenshot = max_msg_per_screenshot
        self.long_examples = set(
            json.load(open(join(self.data_path, "long_examples.json"), "r"))
        )
        self.style = style

        gpt_data_path = join(self.data_path, "gpt5_outputs_all_processed.json")
        self.gpt_data = json.load(open(gpt_data_path, "r"))
        logging.warning(
            f"Loaded {len(self.gpt_data)} GPT data entries from {gpt_data_path}"
        )
        super().__init__(split=self.split)

    def load(self):
        formatted_data = []
        total_skipped_examples = 0
        msg_cnt_dict = {}
        total_screenshots = 0
        for website in tqdm(self.ALL_WEBSITES):
            file_path = join(self.data_path, f"{self.split}_{website}.json")
            if not os.path.exists(file_path):
                print(f"{file_path} not found")
                continue
            with open(file_path, "r") as f:
                data = json.load(f)

            for screenshot in data:
                total_screenshots += 1
                if "viewport" in screenshot:
                    image_w = screenshot["viewport"]["width"]
                    image_h = screenshot["viewport"]["height"]
                else:
                    continue

                screenshot_id = f"{website}__{screenshot['traj_id']}__{screenshot['step_id']}"

                unique_id = f"{screenshot['traj_id']}_{screenshot['step_id']}"
                if unique_id in self.long_examples:
                    continue

                gpt_bid_map = self.gpt_data.get(screenshot_id)
                if gpt_bid_map is None:
                    continue
                msgs = []
                for elem in screenshot["elements"]:
                    if elem.get("bbox", None) is None:
                        continue

                    clickable = elem.get("clickable", False)
                    if self.clickable_only and not clickable:
                        continue

                    x1, y1, w, h = elem["bbox"]
                    bbox = [x1, y1, x1 + w, y1 + h]
                    coords = get_click_coords_from_bbox(bbox, mode=self.mode)
                    if coords[0] > image_w or coords[1] > image_h:
                        total_skipped_examples += 1
                        continue
                    # normalize the coordinates to [0, 100), and round to 1 decimal places
                    normalized_coords = normalize_click_coords(
                        coords[0], coords[1], image_w, image_h
                    )
                    norm_x, norm_y = normalized_coords
                    norm_bbox = normalize_click_coords(
                        bbox[0], bbox[1], image_w, image_h
                    ) + normalize_click_coords(
                        bbox[2], bbox[3], image_w, image_h
                    )
                    elem["bbox"] = norm_bbox
                    if clickable:
                        action = {
                            "name": "click",
                            "button": "left",  # default button
                            "click_type": "single",  # default click type
                            "x": norm_x,
                            "y": norm_y,
                        }
                    else:
                        p = random.random()
                        if p < 0.5 or self.split == "val":
                            # 50% chance to hover, 50% chance to send message for train
                            # but always hover for validation
                            action = {"name": "hover", "x": norm_x, "y": norm_y}
                        else:
                            action = {
                                "name": "send_msg_to_user",
                                "msg": f"Found element {elem['name']} at x={norm_x}, y={norm_y}",
                            }

                    gpt_entry = gpt_bid_map.get(str(elem["bid"]))
                    if gpt_entry is None:
                        total_skipped_examples += 1
                        continue
                    question = gpt_entry["query"]
                    thought = gpt_entry["thought"]
                    if self.action_only:
                        output = action
                    else:
                        output = {
                            "thought": thought,
                            "action": action,
                        }
                    msg = {
                        "question": question,
                        "answer": json.dumps(output),
                        "style": self.style,
                        "task_description": question,
                    }
                    msgs.append(msg)
                    if self.flatten:
                        formatted_example = {
                            "image": screenshot["image_path"],
                            "question": question,
                            "task_description": question,
                            "answer": json.dumps(output),
                            "style": self.style,
                            "metadata": {
                                "traj_id": screenshot["traj_id"],
                                "dataset": website,
                                "step_id": screenshot["step_id"],
                                "url": screenshot["url"],
                                "bbox": norm_bbox,
                                "image_w": image_w,
                                "image_h": image_h,
                            },
                        }
                        formatted_data.append(formatted_example)

                if not self.flatten and len(msgs) > 0:
                    msg_cnt_dict[len(msgs)] = msg_cnt_dict.get(len(msgs), 0) + 1

                    formatted_example = {
                        "image": screenshot["image_path"],
                        "message_list": msgs,
                        "metadata": {
                            "traj_id": screenshot["traj_id"],
                            "dataset": website,
                            "step_id": screenshot["step_id"],
                            "url": screenshot["url"],
                            "image_w": image_w,
                            "image_h": image_h,
                        },
                    }

                    if (
                        self.max_msg_per_screenshot > 0
                        and len(msgs) > self.max_msg_per_screenshot
                    ):
                        flat = []
                        for _msgs in split_into_groups(
                            msgs, self.max_msg_per_screenshot
                        ):  
                            flat.append(
                                dict(formatted_example, message_list=_msgs)
                            )
                        formatted_data.extend(flat)
                    else:
                        formatted_data.append(formatted_example)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        return example


def _process_ground_cua_example(args):
    """
    Module-level helper function for multiprocessing in GroundCUA.load().
    Must be at module level to be picklable.
    """
    ex, mode, web_data_home = args
    image_path = join(web_data_home, "GroundCUA", "GroundCUA/images", ex['image_path'])
    if not exists(image_path):
        return None, "missing_image", f"Image not exists: {image_path}"

    try:
        image_w, image_h = Image.open(image_path).size
    except Exception as e:
        return None, "image_error", f"Failed to open image {image_path}: {e}"

    bbox = ex["bbox"]
    coords = get_click_coords_from_bbox(bbox, mode=mode)

    if coords[0] > image_w or coords[1] > image_h:
        return None, "oob", f"Coordinates OOB: {coords} for image with w={image_w}, h={image_h}"

    # normalize the coordinates to [0, 100), and round to 1 decimal places
    normalized_coords = normalize_click_coords(
        coords[0], coords[1], image_w, image_h
    )
    norm_x, norm_y = normalized_coords
    norm_bbox = normalize_click_coords(
        bbox[0], bbox[1], image_w, image_h
    ) + normalize_click_coords(
        bbox[2], bbox[3], image_w, image_h
    )

    if len(ex['text'].strip()) == 0:
        return None, "empty_text", None

    action = {
        "name": "click",
        "button": "left",
        "click_type": "single",
        "x": norm_x,
        "y": norm_y,
    }

    elem_description = format_elem_description(
        elem_content=ex['text'],
    )
    question = random.choice(WEB_GROUNDING_TEMPLATES).format(
        description=elem_description,
    )
    formatted_item = dict(
        image=image_path,
        question=question,
        task_description=question,
        answer=json.dumps(action),
        style="web_grounding",
        metadata=dict(
            bbox=list(norm_bbox),
            category=ex["category"],
            id=ex["id"]
        )
    )
    return formatted_item, "success", None


class GroundCUA(DatasetBase):
    @classmethod
    def download(cls, n_procs=1, mode="random_gaussian", check_sha=False, n_val=1024, cache_only=False):
        """
        Download and pre-process GroundCUA dataset.

        Args:
            n_procs: Number of processes for parallel processing
            mode: Click coordinate mode ("center", "top_left", "random_gaussian", "random_uniform")
            check_sha: Unused, kept for API compatibility
            n_val: Unused, kept for API compatibility
            cache_only: Unused, kept for API compatibility
        """
        local_name = join(WEB_DATA_HOME, "GroundCUA")
        formatted_data_path = join(local_name, "formatted_data.json")

        # Check if formatted data already exists
        if exists(formatted_data_path):
            print(f"Formatted data already exists at {formatted_data_path}")
            return

        # Create directory if needed
        os.makedirs(local_name, exist_ok=True)

        # Try to load from local disk first, otherwise download from HuggingFace
        local_raw_data_path = join(local_name, "train")
        if exists(local_raw_data_path):
            print(f"Loading raw data from local disk: {local_raw_data_path}")
            ds = datasets.load_from_disk(local_name)["train"]
            data_list = [ex for ex in tqdm(ds, desc="Loading raw data")]
        else:
            print("Downloading GroundCUA dataset from HuggingFace...")
            ds = datasets.load_dataset(
                "ServiceNow/GroundCUA",
                split="train",
                streaming=True
            )
            data_list = []
            for sample in tqdm(ds, desc="Downloading"):
                data_list.append(sample)

        # Process all examples
        print(f"Processing {len(data_list)} examples with {n_procs} processes...")
        args_list = [(ex, mode, WEB_DATA_HOME) for ex in data_list]

        formatted_data = []
        empty_text_cnt = 0
        oob_cnt = 0
        missing_image_cnt = 0
        image_error_cnt = 0

        if n_procs > 1:
            with mp.Pool(n_procs) as pool:
                results = pool.map(_process_ground_cua_example, args_list)

            for result, status, msg in tqdm(results, desc="Collecting results"):
                if status == "success":
                    formatted_data.append(result)
                elif status == "empty_text":
                    empty_text_cnt += 1
                elif status == "oob":
                    oob_cnt += 1
                elif status == "missing_image":
                    missing_image_cnt += 1
                elif status == "image_error":
                    image_error_cnt += 1
        else:
            for args in tqdm(args_list, desc="Processing"):
                result, status, msg = _process_ground_cua_example(args)
                if status == "success":
                    formatted_data.append(result)
                elif status == "empty_text":
                    empty_text_cnt += 1
                elif status == "oob":
                    oob_cnt += 1
                elif status == "missing_image":
                    missing_image_cnt += 1
                elif status == "image_error":
                    image_error_cnt += 1

        print(f"Processed {len(formatted_data)} examples")
        print(f"Skipped: {empty_text_cnt} empty text, {oob_cnt} OOB, {missing_image_cnt} missing images, {image_error_cnt} image errors")

        # Save formatted data to JSON
        print(f"Saving formatted data to {formatted_data_path}...")
        with open(formatted_data_path, "w") as f:
            json.dump({"data": formatted_data, "mode": mode}, f)

        print("Download and processing complete!")

    def __init__(
            self, 
            split: Literal["train"] = "train",
            flatten=False, 
            max_msg_per_screenshot=-1
        ):
        """
        Initialize GroundCUA dataset.

        Args:
            split: Dataset split (only "train" is available)
            flatten: If True, return individual examples per element.
                     If False, merge examples for the same image into message_list.
            max_msg_per_screenshot: Maximum messages per screenshot when not flattened.
                                    -1 means no limit.
        """
        self.split = split
        self.flatten = flatten
        self.max_msg_per_screenshot = max_msg_per_screenshot
        self.formatted_data_path = join(WEB_DATA_HOME, "GroundCUA", "formatted_data.json")
        super().__init__(split)

    def load(self):
        """Load pre-processed formatted data from disk."""
        if not exists(self.formatted_data_path):
            raise FileNotFoundError(
                f"Formatted data not found at {self.formatted_data_path}. "
                "Please run GroundCUA.download() first."
            )

        print(f"Loading pre-processed GroundCUA data from {self.formatted_data_path}...")
        with open(self.formatted_data_path, "r") as f:
            saved_data = json.load(f)

        raw_data = saved_data["data"]
        mode = saved_data.get("mode", "unknown")
        logging.info(f"Loaded {len(raw_data)} raw examples (mode: {mode})")

        if self.flatten:
            # Return individual examples (original behavior)
            return raw_data

        # Merge examples for the same image into message_list (like SyntheticGround)
        image_to_examples = {}
        for ex in raw_data:
            image_path = ex["image"]
            if image_path not in image_to_examples:
                image_to_examples[image_path] = {
                    "image": image_path,
                    "message_list": [],
                    "metadata": ex.get("metadata", {}),
                }
            msg = {
                "question": ex["question"],
                "answer": ex["answer"],
                "style": ex.get("style", "web_grounding"),
                "task_description": ex.get("task_description", ex["question"]),
            }
            image_to_examples[image_path]["message_list"].append(msg)

        formatted_data = []
        for image_path, example in image_to_examples.items():
            msgs = example["message_list"]
            if self.max_msg_per_screenshot > 0 and len(msgs) > self.max_msg_per_screenshot:
                # Split into multiple examples if too many messages
                for _msgs in split_into_groups(msgs, self.max_msg_per_screenshot):
                    formatted_data.append(dict(example, message_list=_msgs))
            else:
                formatted_data.append(example)

        logging.info(f"Merged into {len(formatted_data)} examples (flatten={self.flatten})")
        return formatted_data

    def get(self, item, rng):
        return self.data[item]
