from typing import Optional, Union
from pathlib import Path
import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import GPT2, GPTJ, T0
import clip
from create_dataset import CocoCaptionDataset, CocoCaptionDatasetBase, CocoImageDataset, FolderCaptionDataset, FolderImageDataset
from evaluate_model import ClipGuidedCaptionSampler, ClipScoring, CocoCaptionValidator, NoBeamCaptionSampler


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
        if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
            output_path = self.output_path / f"{self.filename_prefix}_epoch_{epoch}{'.ckpt' if not self.use_deepspeed else ''}"
            trainer.save_checkpoint(output_path)

    def on_batch_end(self, trainer: pl.Trainer, _):
        if self.save_every_n_steps is not None:
            current_step = trainer.global_step
            if self.save_every_n_steps > 0 and current_step % self.save_every_n_steps == 0:
                output_path = self.output_path / f"{self.filename_prefix}_latest{'.ckpt' if not self.use_deepspeed else ''}"
                trainer.save_checkpoint(output_path)

    def save_final_checkpoint(self, trainer: pl.Trainer):
        output_path = self.output_path / f"{self.filename_prefix}_final{'.ckpt' if not self.use_deepspeed else ''}"
        trainer.save_checkpoint(output_path)


def train(
    input_dataset: str=None,     # path of COCO train annotation json file
    image_folder_path: str=None,
    valid_json_path: str=None,   # path of COCO valid annotation json file
    valid_image_folder_path: str=None,
    validation_interval:Union[int, float] = 1000,
    max_token_length: int = 96,
    output_dir: str = "./models/",
    output_name_prefix: str = "demo_model.ckpt",
    epochs: int = 3,
    save_every_epochs: int = 1,
    save_every_steps: int = 10000,
    scheduler_warmup_steps: int = 2000,
    prefix_length: int = 10,
    prefix_size: int = 768,
    clip_prefix_length: int = 50,       # e.g. reduce to 10 when not using all vit-features
    pos_embeddings: bool = False,       # learn position embedding in mapping transformer
    language_model_type = "gpt2",
    language_model_variant = "gpt2",    # "gpt2-xl"
    visual_encoder_type: str = 'BLIP',  # BLIP or CLIP
    visual_encoder_model_variant: str = 'ViT-B',
    batch_size: int = 16,
    optimizer_lr: float = 2e-5,
    prefix_only: bool = False,
    use_all_vit_features: bool = True,
    num_layers: int = 8,
    num_attention_heads: int = 8,
    mlp_ratio: float=4.,
    prefix_init_std: float=1.,
    act_fn_name: str='relu',    # activation function used for transformer mapper
    use_deepspeed: bool = False,
    use_wandb: bool = False,
    wandb_project: str="CLIP-Image-Captioning",
    wandb_name: str=None,
    log_every_n_steps: int = 5,
    use_16bit_precision: bool = True,
    gpu_devices: Optional[str] = "0",
    deepspeed_strategy: Optional[str] = None,
    replace_extension: str = None,
    resize_transform: bool = True,
    num_workers: int=8,
    max_log_samples: int=64,
    autoclip_p: int=10,
    enable_checkpointing: bool=False,
    acc_grad_batches: int=1,
    gradient_checkpointing_enable: bool=True
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

    tokenizer = CocoCaptionDatasetBase.create_tokenizer(tokenizer_model_type=language_model_type, tokenizer_model_variant=language_model_variant)

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

    
    if input_dataset is not None:
        dataset = CocoCaptionDataset(
            annotation_json_path=input_dataset,
            image_folder_path=image_folder_path,
            image_transform=preprocess,
            tokenizer=tokenizer,
            max_token_length=max_token_length,
            replace_extension=replace_extension
        )
    elif image_folder_path is not None:
        dataset = FolderCaptionDataset(
            image_folder_path,
            tokenizer=tokenizer,
            image_transform=preprocess,
            max_token_length=max_token_length
        )
    else:
        raise RuntimeError('Neither input_dataset nor image_folder_path was specified.')

    total_steps = len(dataset) // batch_size * epochs

    model_kwargs = {
        "language_model_type": language_model_type,
        "language_model_variant": language_model_variant,
        "prefix_length": prefix_length,
        "clip_prefix_length": clip_prefix_length,
        "prefix_size": prefix_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "mlp_ratio": mlp_ratio,
        "prefix_init_std": prefix_init_std,
        "act_fn_name": act_fn_name,
        "use_all_vit_features": use_all_vit_features,
        "pos_embeddings": pos_embeddings,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "total_steps": total_steps,
        "use_deepspeed": use_deepspeed,
        "optimizer_lr": optimizer_lr,
        "validation_interval": validation_interval,
        "max_log_samples": max_log_samples
    }

    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant, use_cache=False)
    elif language_model_type in ("gptj", "gpt-j"):
        language_model = GPTJ.create(language_model_variant)
    elif language_model_type in ("t0", "t5"):
        language_model = T0.create(language_model_variant)
    else:
        raise ValueError(f"invalid language model type '{language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")

    if gradient_checkpointing_enable:
        language_model.gradient_checkpointing_enable()

    # prepare model validator
    val_clip_model = "ViT-B/32"
    clip_model, clip_image_preprocess = clip.load(val_clip_model, device=device, jit=False)
    if valid_json_path is not None:
        validation_dataset = CocoImageDataset(annotation_json_path=valid_json_path, image_folder_path=valid_image_folder_path, replace_extension=replace_extension)
    elif valid_image_folder_path is not None:
        validation_dataset = FolderImageDataset(valid_image_folder_path)
    else:
        validation_dataset = None

    def validation_collate(batch):
        """directly pass on PIL images"""
        return batch

    if validation_dataset is not None:
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=validation_collate, pin_memory=False)
        nobeam_sampler = NoBeamCaptionSampler(top_p_values=[0.1, 0.2])
        clip_scoring = ClipScoring(clip_model, clip_image_preprocess)
        clip_guided_sampler = ClipGuidedCaptionSampler(clip_scoring, branching_factor=2, look_ahead=4)
        validator = CocoCaptionValidator(
            validation_dataset,
            preprocess,
            {
                'nobeam': nobeam_sampler,
                'clip_guided': clip_guided_sampler
            },
            clip_scoring
        )
    else:
        validator = None
        validation_dataloader = None

    if prefix_only:
        language_model = language_model.eval()
        for param in language_model.parameters():
            param.requires_grad = False

        model = CLIPCaptionPrefixOnly(language_model, tokenizer, encode_image, validator=validator, autoclip_p=autoclip_p, **model_kwargs)
        print("Train only Prefix.")
    else:
        model = CLIPCaptionModel(language_model, tokenizer, encode_image, validator=validator, autoclip_p=autoclip_p, **model_kwargs)
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
        logger = WandbLogger(project=wandb_project, name=wandb_name, log_model=False)
    else:
        logger = None

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

    callbacks=[checkpoint_saver]

    if logger is not None:
        lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False)
        callbacks.append(lr_monitor)

    # Create trainer class.
    trainer = pl.Trainer(
        gpus=gpu_devices,
        max_epochs=epochs,
        callbacks=callbacks,
        strategy=deepspeed_strategy,
        precision=(16 if use_16bit_precision else 32),
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=-1,    # use adaptive gradient clipping, configure_gradient_clipping is overwritten
        val_check_interval=validation_interval,
        check_val_every_n_epoch=1,
        limit_val_batches=100,
        enable_checkpointing=enable_checkpointing,
        accumulate_grad_batches=acc_grad_batches
    )

    # Run training process.
    trainer.fit(model, dataloader, validation_dataloader)

    # Save final checkpoint.
    checkpoint_saver.save_final_checkpoint(trainer)


if __name__ == '__main__':
    fire.Fire(train)