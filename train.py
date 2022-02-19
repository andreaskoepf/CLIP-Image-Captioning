from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import fire

from dataset import TokenPrefixDataset, MultiplePrefixDataset
from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import GPT2, GPTJ, T0

from create_dataset import CocoCaptionDataset


class CheckpointSaver(pl.Callback):
    def __init__(self, output_path: Path, filename_prefix: str, save_every_n_epochs: int = 1,
            save_every_n_steps: Optional[int] = 1000, use_deepspeed: bool = False):
        output_path.mkdir(exist_ok=True)

        self.use_deepspeed = use_deepspeed
        self.output_path = output_path
        self.filename_prefix = filename_prefix
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps

    def on_epoch_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch
        if epoch % self.save_every_n_epochs == 0:
            output_path = self.output_path / f"{self.filename_prefix}_epoch_{epoch}{'.ckpt' if not self.use_deepspeed else ''}"
            trainer.save_checkpoint(output_path)
    
    def on_batch_end(self, trainer: pl.Trainer, _):
        if self.save_every_n_steps is not None:
            current_step = trainer.global_step
            if (current_step % self.save_every_n_steps == 0):
                output_path = self.output_path / f"{self.filename_prefix}_latest{'.ckpt' if not self.use_deepspeed else ''}"
                trainer.save_checkpoint(output_path)
    
    def save_final_checkpoint(self, trainer: pl.Trainer):
        output_path = self.output_path / f"{self.filename_prefix}_final{'.ckpt' if not self.use_deepspeed else ''}"
        trainer.save_checkpoint(output_path)


def train(
    input_dataset: str,     # path of COCO annotation json file
    image_folder_path: str,
    max_token_length: int = 96,
    #data_dir: str = "./train/",
    output_dir: str = "./models/",
    output_name_prefix: str = "demo_model.ckpt",
    epochs: int = 3,
    save_every_epochs: int = 1,
    save_every_steps: int = 10000,
    scheduler_warmup_steps: int = 2000,
    prefix_length: int = 10,
    prefix_size: int = 768,
    clip_prefix_length: int = 50,       # e.g. reduce to 10 when not using all vit-features
    pos_embeddings: bool = False,        # learn position embedding in mapping transformer
    language_model_type = "gpt2",
    language_model_variant = "gpt2-xl",
    visual_encoder_type: str = 'BLIP',  # BLIP or CLIP
    visual_encoder_model_variant: str = 'ViT-B',
    batch_size: int = 16,
    optimizer_lr: float = 2e-5,
    prefix_only: bool = False,
    use_all_vit_features: bool = True,
    num_layers: int = 8,
    num_attention_heads: int = 8,
    normalize_prefix: bool = False,
    merge_datasets: bool = False,
    use_deepspeed: bool = False,
    use_wandb: bool = False,
    log_every_n_steps: int = 50,
    use_16bit_precision: bool = True,
    gpu_devices: Optional[str] = "0",
    deepspeed_strategy: Optional[str] = None,
    replace_extension: str = None,
    resize_transform: bool = True
):
    """ Starts the main training process. """ # TODO arg docs.

    print(f'Using pytorch version {torch.__version__}')
    print('Args: ', locals())

    # Easier to use GPU args. `-1` = use all, `0` = use gpu 0, `0,1` = use gpus 1 and 2 etc.
    if isinstance(gpu_devices, str):
        gpu_devices = [int(x) for x in gpu_devices.split(',')]
    elif isinstance(gpu_devices, int) and gpu_devices != -1:
        gpu_devices = [gpu_devices]
    if gpu_devices[0] == -1:
        gpu_devices = -1

    if visual_encoder_type == 'BLIP':
        if visual_encoder_model_variant == 'ViT-B':
            blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        else:
            raise RuntimeError('Visual encoder model variant not supported: \'{visual_encoder_model_variant}\'')

        image_size = 384
        if resize_transform:
            preprocess = transforms.Compose([
                transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            print('BLIP transform without resize: Expecting correctly sized input images.')
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        from BLIP.models.blip import blip_decoder

        blip_model = blip_decoder(pretrained=blip_model_url, image_size=image_size, vit='base', med_config='BLIP/configs/med_config.json')
        blip_model.eval()

        device = torch.device('cuda', max(gpu_devices if isinstance(gpu_devices, int) else gpu_devices[0], 0))
        blip_model = blip_model.to(device)

        def blip_encode(image):
            with torch.no_grad():
                return blip_model.visual_encoder(image)

        encode_image = blip_encode
    else:
        raise RuntimeError('Unsupported visual encdore \'{visual_encoder_type}\' specified.')

    tokenizer = CocoCaptionDataset.create_tokenizer(tokenizer_model_type=language_model_type, tokenizer_model_variant=language_model_variant)
    dataset = CocoCaptionDataset(annotation_json_path=input_dataset, image_folder_path=image_folder_path, image_transform=preprocess, 
        tokenizer=tokenizer, max_token_length=max_token_length, replace_extension=replace_extension)

    # # Prepare training datasets.
    # if merge_datasets:
    #     data_dirs = data_dir.split(",")

    #     if len(data_dirs) < 2:
    #         raise ValueError(
    #             "--merge_datasets was enabled, but less than 2 directories were specified.\n"
    #             "You can specify more than one data directory by comma seperating the --data_dir input."
    #         )
        
    #     datasets = []
    #     for dir in data_dirs:
    #         datasets.append(
    #             TokenPrefixDataset(dir, batch_size=batch_size, normalize_prefix=normalize_prefix)
    #         )
        
    #     dataset = MultiplePrefixDataset(*datasets)
    # else:
    #     dataset = TokenPrefixDataset(data_dir, batch_size=batch_size, normalize_prefix=normalize_prefix)

    # TODO find better solution for using `get_linear_schedule_with_warmup` with PL.
    total_steps = len(dataset) * epochs # batch size is already accounted for in `len(dataset)`

    model_kwargs = {
        "language_model_type": language_model_type,
        "language_model_variant": language_model_variant,
        "prefix_length": prefix_length,
        "clip_prefix_length": clip_prefix_length,
        "prefix_size": prefix_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "use_all_vit_features": use_all_vit_features,
        "pos_embeddings": pos_embeddings,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "total_steps": total_steps,
        "use_deepspeed": use_deepspeed,
        "optimizer_lr": optimizer_lr
    }
    
    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant)
    elif language_model_type in ("gptj", "gpt-j"):
        language_model = GPTJ.create(language_model_variant)
    elif language_model_type in ("t0", "t5"):
        language_model = T0.create(language_model_variant)
    else:
        raise ValueError(f"invalid language model type '{language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")
    
    if prefix_only:
        language_model = language_model.eval()
        for param in language_model.parameters():
            param.requires_grad = False 

        model = CLIPCaptionPrefixOnly(language_model, encode_image, **model_kwargs)
        print("Train only Prefix.")
    else:
        model = CLIPCaptionModel(language_model, encode_image, **model_kwargs)
        print("Train both Prefix and Language Model.")
    
    # Create `CheckpointSaver` as a trainer callback instance.
    checkpoint_saver = CheckpointSaver(
        Path(output_dir),
        output_name_prefix,
        save_every_n_epochs=save_every_epochs,
        save_every_n_steps=save_every_steps,
        use_deepspeed=use_deepspeed
    )

    if use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="CLIP-Image-Captioning")
    else:
        logger = None
    
    # TODO better dataset implementation
    # - Improve dataloader system (batch_size=1 is a temporary fix)
    # - Speed up streaming (multiple workers and/or prepare data ahead of retrieval)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create trainer class.
    trainer = pl.Trainer(
        gpus=gpu_devices,
        max_epochs=epochs,
        callbacks=[checkpoint_saver],
        strategy=deepspeed_strategy,
        precision=(16 if use_16bit_precision else 32),
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        #track_grad_norm=2,
    )

    # Run training process.
    trainer.fit(model, dataloader)

    # Save final checkpoint.
    checkpoint_saver.save_final_checkpoint(trainer)


if __name__ == '__main__':
    fire.Fire(train)