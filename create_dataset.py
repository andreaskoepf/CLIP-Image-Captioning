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
            self.tokenizer.encode_text(caption),
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
            self.tokenizer.encode_text(caption),
            dtype=torch.int64
        )

        padding = self.max_token_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_token_length]
        
        output["tokens"] = tokens.numpy()

        return output


def create_webdataset(
    urls,
    image_transform,
    image_key: str = "jpg",
    caption_key: str = "txt",
    tokenizer_model_type: str = "gpt2",
    tokenizer_model_variant: str = "gpt2-xl",
    caption_in_metadata: bool = False,
    max_token_length: int = 128
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import webdataset as wds

    dataset = wds.WebDataset(urls, handler=wds.handlers.warn_and_continue)
    
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

    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def filter_dataset(item):
        if caption_key not in item and not caption_in_metadata:
            return False
        if image_key not in item:
            return False
        if caption_in_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}

        image_data = item[image_key]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = image_transform(image)
        output["image_tensor"] = image_tensor

        if not caption_in_metadata:
            text = item[caption_key]
            caption = text.decode("utf-8")
        else:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            caption = json.loads(metadata)[caption_key]
        

        # question = either VQA question or None.
        # answer = either raw caption or VQA answer.

        tokens = torch.tensor(
            tokenizer.encode_text(caption),
            dtype=torch.int64
        )
        
        padding = max_token_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:max_token_length]
        
        output["tokens"] = tokens.numpy()
        
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, write_batch_size):
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.prefixes_folder = output_folder + "/prefixes"
        self.tokens_folder = output_folder + "/tokens"

        if not self.fs.exists(self.output_folder):
            self.fs.mkdir(self.output_folder)
            batch_init_num = -1
        else:
            existing_top_level_files = self.fs.walk(self.prefixes_folder).__next__()[2]
            if len(existing_top_level_files) == 0:
                batch_init_num = -1
            else:
                batch_init_num = max(
                    [int(x.split("/")[-1].split(".")[0].split("_")[1]) for x in existing_top_level_files]
                )
        
        if not self.fs.exists(self.prefixes_folder):
            self.fs.mkdir(self.prefixes_folder)

        if not self.fs.exists(self.tokens_folder):
            self.fs.mkdir(self.tokens_folder)

        self.write_batch_size = write_batch_size
        self.batch_count = 0
        self.batch_num = batch_init_num
        self.__init_batch()

    def __init_batch(self):
        self.prefixes = []
        self.tokens = []
        self.batch_count = 0
        self.batch_num += 1

    def add(self, prefixes, tokens):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """
        self.batch_count += prefixes.shape[0]
        self.prefixes.append(prefixes)
        self.tokens.append(tokens)

        if self.batch_count > self.write_batch_size:
            self.flush()

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """

        img_emb_mat = np.concatenate(self.prefixes)
        output_path_img = self.prefixes_folder + "/prefixes_" + str(self.batch_num)

        with self.fs.open(output_path_img + ".npy", "wb") as f:
            npb = BytesIO()
            np.save(npb, img_emb_mat)
            f.write(npb.getbuffer())

        tokens_mat = np.concatenate(self.tokens)
        output_path_text = self.tokens_folder + "/tokens_" + str(self.batch_num)

        with self.fs.open(output_path_text + ".npy", "wb") as f:
            npb = BytesIO()
            np.save(npb, tokens_mat)
            f.write(npb.getbuffer())

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


def preprocess_dataset(
    input_dataset: str,
    output_folder: str,
    input_format: str = "files",
    batch_size: int = 256,
    num_prepro_workers: int = 8,
    write_batch_size: int = (10 ** 6),
    subset_size: Optional[int] = None,
    wds_image_key: Optional[str] = None,
    wds_caption_key: Optional[str] = None,
    wds_caption_in_metadata: bool = False,
    wds_vqa_question_key: Optional[str] = None,
    visual_encoder_type: str = 'BLIP',  # BLIP or CLIP
    visual_encoder_model_variant: str = 'ViT-B',
    clip_model: str = "ViT-B/32",
    tokenizer_model_type: str = "gpt2",
    tokenizer_model_variant: str = "gpt2-xl",
    max_token_length: int = 128,
    use_all_vit_features: bool = True,
    device: str = "cuda:0",
    image_folder_path: str = None
):
    encode_image = None

    if visual_encoder_type == 'CLIP':
        model, preprocess = clip.load(clip_model, device=device, jit=False)

        if use_all_vit_features:
            def vit_forward_patch(self, x: torch.Tensor):
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
                x = self.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

                # this patch removes the CLS token output extraction + projection from CLIP's ViT forward method
                # original: https://github.com/openai/CLIP/blob/40f5484c1c74edd83cb9cf687c6ab92b28d8b656/clip/model.py#L202-L236

                #x = self.ln_post(x[:, 0, :])

                if self.proj is not None:
                    x = x @ self.proj

                return x

            model.visual.forward = vit_forward_patch.__get__(model.visual, VisionTransformer)
        
        encode_image = lambda img: model.encode_image(img)

    elif visual_encoder_type == 'BLIP':
        if visual_encoder_model_variant == 'ViT-B':
            blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        else:
            raise RuntimeError('Visual encoder model variant not supported: \'{visual_encoder_model_variant}\'')

        image_size = 384
        preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        from BLIP.models.blip import blip_decoder

        model = blip_decoder(pretrained=blip_model_url, image_size=image_size, vit='base', med_config='BLIP/configs/med_config.json')
        model.eval()
        model = model.to(device)

        encode_image = lambda img: model.visual_encoder(img)
    else:
        raise RuntimeError('Unsupported visual encdore \'{visual_encoder_type}\' specified.')

    if input_format == "files":
        dataset = FileFolderDataset(
            preprocess,
            input_dataset,
            tokenizer_model_type=tokenizer_model_type,
            tokenizer_model_variant=tokenizer_model_variant,
            max_token_length=max_token_length
        )
    elif input_format == "webdataset":
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            caption_in_metadata=wds_caption_in_metadata,
            wds_vqa_question_key=wds_vqa_question_key,
            tokenizer_model_type=tokenizer_model_type,
            tokenizer_model_variant=tokenizer_model_variant,
            max_token_length=max_token_length
        )
    elif input_format == "coco_json":
        tokenizer = CocoCaptionDataset.create_tokenizer(tokenizer_model_type=tokenizer_model_type, tokenizer_model_variant=tokenizer_model_variant)
        dataset = CocoCaptionDataset(annotation_json_path=input_dataset, image_folder_path=image_folder_path, image_transform=preprocess, tokenizer=tokenizer, max_token_length=max_token_length)
    else:
        raise Exception(f"No such input format {input_format}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format in ("files", "coco_json") else None,
    )
    output_sink = OutputSink(output_folder, write_batch_size)

    c = 0
    bar = tqdm.tqdm()
    for items in data:
        with torch.no_grad():
            image = items["image_tensor"].to(device)
            image_embs = encode_image(image).cpu().numpy()

            tokens = items["tokens"]

            output_sink.add(image_embs, tokens)

        bar.update(batch_size)
        c += batch_size
        if subset_size is not None and c >= subset_size:
            break
        
    output_sink.flush()


if __name__ == "__main__":
    fire.Fire(preprocess_dataset)