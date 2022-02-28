""" A modified version of clip_inference.py from rom1504/clip-retrieval """
from dataclasses import dataclass
import re
from typing import Optional
from pathlib import Path
from io import BytesIO
import io
import json

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_tensor

from PIL import Image, UnidentifiedImageError
import numpy as np
import fsspec
import tqdm
import fire

import clip
from clip.model import VisionTransformer


@dataclass
class CocoJsonImageEntry:
    id: str
    file_name: str
    url: str


@dataclass
class CocoJsonCaptionEntry:
    caption: str
    image: CocoJsonImageEntry


class DatasetIndexBase(Dataset):
    def get_captions_by_image_id(self):
        captions = {}
        for entry in self:
            if entry.image.id in captions:
                captions[entry.image.id].append(entry.caption)
            else:
                captions[entry.image.id] = [entry.caption]
        return captions
    
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index: int):
        raise NotImplementedError()


class CocoJsonDataset(DatasetIndexBase):
    """
    Non-standard dataset providing CocoJsonCaptionEntry objects. It is not used directly but aggregated by
    other Dataset classes (CocoImageDataset, CocoCaptionDataset) to access COCO captions read from the COCO
    json annotation files.
    """
    def __init__(self, annotation_json_path):
        super().__init__()

        with open(annotation_json_path, "r") as f:
            j = json.load(f)

        images = j["images"]
        image_by_id = dict()
        for img in images:
            url = img['coco_url'] if 'coco_url' in img else None
            image_by_id[img['id']] = CocoJsonImageEntry(id=img['id'], file_name=img['file_name'], url=url)
        self.image_by_id = image_by_id
        self.annotations = j["annotations"]
        print(f'total annotations: {len(self.annotations)}; total images: {len(image_by_id)};')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        a = self.annotations[index]

        caption = a['caption']
        image_id = a['image_id']
        image = self.image_by_id[image_id]

        return CocoJsonCaptionEntry(caption=caption, image=image)


class FileFolderIndexDataset(DatasetIndexBase):
    def __init__(self, folder_path):
        super().__init__()

        path = Path(folder_path)

        text_files = { fn.stem: fn for fn in path.glob("**/*.txt") }
        
        image_files = [
            *path.glob("**/*.png"),
            *path.glob("**/*.jpg"),
            *path.glob("**/*.jpeg"),
            *path.glob("**/*.bmp"),
        ]
        
        image_files = { fn.stem: fn for fn in image_files }
        keys = text_files.keys() & image_files.keys()   # intersection between text and image key sets

        self.image_by_id = dict((k, CocoJsonImageEntry(id=k, file_name=v, url=None)) for k,v in image_files.items() if k in keys)
        
        self.keys = list(keys)
        self.text_files = dict((k,v) for k,v in text_files.items() if k in keys)
        self.keys = list(keys)
        
        print(f'total images-text pairs: {len(self.image_by_id)};')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        image = self.image_by_id[key]
        caption = self.text_files[key].read_text()
        caption = re.sub("\s+", " ", caption)
        return CocoJsonCaptionEntry(caption=caption, image=image)


class CocoImageDatasetBase(Dataset):
    """
    Dataset returning image tensors together with image entry objects. Mainly used for evaluating the model.  
    """
    def __init__(self, annotations: DatasetIndexBase, image_folder_path: str, replace_extension: str = None):
        super().__init__()
        self.annotations = annotations
        self.keys = list(self.annotations.image_by_id.keys())
        self.image_folder_path = Path(image_folder_path) if isinstance(image_folder_path, str) else image_folder_path
        self.replace_extension = replace_extension

    def get_index(self):
        return self.annotations

    def __len__(self):
        return len(self.keys)
    
    def get_image_path_by_id(self, image_id):
        image_entry = self.annotations.image_by_id[image_id]
        file_path = image_entry.file_name
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert isinstance(file_path, Path)
        parent_path = self.image_folder_path or file_path.parent
        if self.replace_extension is not None:
            file_path = file_path.stem + self.replace_extension
        return parent_path / file_path

    def load_image_by_id(self, image_id):
        image_path = self.get_image_path_by_id(image_id)
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):
        image_id = self.keys[index]
        image_entry = self.annotations.image_by_id[image_id]

        try:
            image = self.load_image_by_id(self.keys[index])
        except BaseException as err:
            print(f"Failed to load image '{self.get_image_path_by_id()}' (error='{err}'; type(err)={type(err)}). Skipping.")
            return None  # return None to be filtered in the batch collate_fn

        return {
            "image": image,
            "image_entry": image_entry
        }


class CocoImageDataset(CocoImageDatasetBase):
    """
    Dataset returning image tensors together with image entry objects. Mainly used for evaluating the model.  
    """
    def __init__(self, annotation_json_path: str, image_folder_path: str, replace_extension: str = None):
        super().__init__(CocoJsonDataset(annotation_json_path), image_folder_path, replace_extension)


class FolderImageDataset(CocoImageDatasetBase):
    def __init__(self, folder_path: str):
        super().__init__(FileFolderIndexDataset(folder_path), image_folder_path=None)


class CocoCaptionDatasetBase(Dataset):
    def __init__(self, annotations: DatasetIndexBase, image_folder_path: str, tokenizer, image_transform, max_token_length: int = 128, replace_extension: str = None):
        super().__init__()
        self.annotations = annotations
        self.image_folder_path = Path(image_folder_path) if isinstance(image_folder_path, str) else image_folder_path
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.replace_extension = replace_extension

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        entry = self.annotations[index]

        caption = entry.caption

        file_path = entry.image.file_name
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert isinstance(file_path, Path)
        parent_path = self.image_folder_path or file_path.parent
        if self.replace_extension is not None:
            file_path = file_path.stem + self.replace_extension
        image_path = parent_path / file_path

        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform is not None:
                image_tensor = self.image_transform(image)
            else:
                image_tensor = to_tensor(image)
        except BaseException as err:
            print(f"Failed to load image '{image_path}' (error='{err}'; type(err)={type(err)}). Skipping.")
            return None  # return None to be filtered in the batch collate_fn

        tokens = torch.tensor(
            self.tokenizer.encode_text(caption, max_token_length=self.max_token_length, add_bos=True, add_eos=True),
            dtype=torch.int64
        )

        padding = self.max_token_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_token_length]
        
        return {
            "image_tensor": image_tensor,
            "tokens": tokens.numpy(),
            "image_id": entry.image.id
        }

    @staticmethod
    def create_tokenizer(tokenizer_model_type: str = "gpt2", tokenizer_model_variant: str = "gpt2-xl"):
        if tokenizer_model_type == "gpt2":
            from lms import GPT2_Tokenizer
            tokenizer = GPT2_Tokenizer.create(tokenizer_model_variant)
        elif tokenizer_model_type in ("gptj", "gpt-j"):
            from lms import GPTJ_Tokenizer
            tokenizer = GPTJ_Tokenizer.create(tokenizer_model_variant)
        elif tokenizer_model_type in ("t5", "t0"):
            from lms import T0_Tokenizer
            tokenizer = T0_Tokenizer.create(tokenizer_model_variant)
        else:
            raise ValueError(f"invalid tokenizer model type: '{tokenizer_model_type}' (expected gpt2/gpt-j/t0/t5)")
        return tokenizer


class CocoCaptionDataset(CocoCaptionDatasetBase):
    def __init__(self, annotation_json_path: str, image_folder_path: str, tokenizer, image_transform, max_token_length: int = 128, replace_extension: str = None):
        super().__init__(CocoJsonDataset(annotation_json_path), image_folder_path, tokenizer, image_transform, max_token_length, replace_extension)


class FolderCaptionDataset(CocoCaptionDatasetBase):
    def __init__(self, folder_path: str, tokenizer, image_transform, max_token_length: int = 128):
        super().__init__(FileFolderIndexDataset(folder_path), image_folder_path=None, tokenizer=tokenizer, image_transform=image_transform, max_token_length=max_token_length)


class FileFolderDataset(Dataset):

    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""

    def __init__(self,
        preprocess,
        folder,
        tokenizer_model_type: str = "gpt2",
        tokenizer_model_variant: str = "gpt2-xl",
        max_token_length: int = 128
    ):
        super().__init__()

        path = Path(folder)

        text_files = [*path.glob("**/*.txt")]
        text_files = {text_file.stem: text_file for text_file in text_files}
        
        image_files = [
            *path.glob("**/*.png"),
            *path.glob("**/*.jpg"),
            *path.glob("**/*.jpeg"),
            *path.glob("**/*.bmp"),
        ]
        
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = None
        join = lambda new_set: new_set & keys if keys is not None else new_set
        
        keys = join(text_files.keys())
        keys = join(image_files.keys())

        self.keys = list(keys)
        
        if tokenizer_model_type == "gpt2":
            from lms import GPT2_Tokenizer
            tokenizer = GPT2_Tokenizer.create(tokenizer_model_variant)
        elif tokenizer_model_type in ("gptj", "gpt-j"):
            from lms import GPTJ_Tokenizer
            tokenizer = GPTJ_Tokenizer.create(tokenizer_model_variant)
        elif tokenizer_model_type in ("t5", "t0"):
            from lms import T0_Tokenizer
            tokenizer = T0_Tokenizer.create(tokenizer_model_variant)
        else:
            raise ValueError(f"invalid tokenizer model type: '{tokenizer_model_type}' (expected gpt2/gpt-j/t0/t5)")

        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        
        self.text_files = {k: v for k, v in text_files.items() if k in keys}

        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.image_transform = preprocess

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        output = {}

        try:
            image_file = self.image_files[key]
            image_tensor = self.image_transform(Image.open(image_file).convert('RGB'))
        except (UnidentifiedImageError, OSError):
            print(f"Failed to load image {image_file}. Skipping.")
            return None  # return None to be filtered in the batch collate_fn

        output["image_tensor"] = image_tensor

        text_file = self.text_files[key]
        caption = text_file.read_text()

        tokens = torch.tensor(
            self.tokenizer.encode_text(caption, max_token_length=self.max_token_length, add_bos=True, add_eos=True),
            dtype=torch.int64
        )

        padding = self.max_token_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_token_length]
        
        output["tokens"] = tokens.numpy()

        return output
