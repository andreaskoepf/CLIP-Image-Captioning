import math
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Union, Tuple
from torch.nn import functional as nnf
import pytorch_lightning as pl
import torch
import wandb

from layers import TransformerMapper, TransformerMapperAllFeatures
from lms import GPT2, GPTJ, T0
from auto_clip import AutoClip


class CaptionValidator:
    def process(self, model, batch):
        pass
    def reset(self):
        pass
    def get_results(self):
        pass
    def load_image_by_id(self, image_id):
        pass


class CLIPCaptionModel(pl.LightningModule):
    def __init__(self,
        language_model: Union[GPT2, GPTJ, T0],
        tokenizer,
        encode_image,
        validator: CaptionValidator, 
        max_log_samples: int=64,
        **kwargs
    ):
        super().__init__()

        # Save hparams (see `train.py` for arguments).
        self.save_hyperparameters(ignore=["language_model"])

        self.language_model = language_model
        self.tokenizer = tokenizer
        self.lm_embedding_size = self.language_model.get_embedding_size()

        self.encode_image = encode_image
        self.autoclip = AutoClip(percentile=10)
        self.validator = validator

        if self.hparams.use_all_vit_features:
            print('Using all ViT features.')
            self.clip_project = TransformerMapperAllFeatures(
                dim_clip=self.hparams.prefix_size,
                dim_embedding=self.lm_embedding_size,
                prefix_length=self.hparams.prefix_length,
                clip_length=self.hparams.clip_prefix_length,
                use_pos_embeddings=self.hparams.pos_embeddings,
                num_heads=self.hparams.num_attention_heads,
                num_layers=self.hparams.num_layers
            )
        else:
            self.clip_project = TransformerMapper(
                dim_clip=self.hparams.prefix_size,
                dim_embedding=self.lm_embedding_size,
                prefix_length=self.hparams.prefix_length,
                clip_length=self.hparams.clip_prefix_length,
                num_heads=self.hparams.num_attention_heads,
                num_layers=self.hparams.num_layers
            )

    def validation_step(self, batch, batch_idx):
        if self.validator is None:
            return -1
        self.validator.process(self, batch)

    def validation_epoch_end(self, val_step_outputs):
        results = self.validator.get_results()
        self.validator.reset()

        self.log_dict({
            'val/loss': results['validation_loss'],
            'val/clip_score': results['clip_score']
        })
        for sampler_id, scores in results['sampler_scores'].items(): 
            self.log_dict({
                f'val/{sampler_id}.CIDEr': scores['CIDEr'],
                f'val/{sampler_id}.Bleu_4': scores['Bleu_4']
            })

        # log validation results to wandb
        max_log_samples = self.hparams.max_log_samples
        thumbnail_size = 128, 128
        if self.logger:
            columns = ["image_id", "image_url", "thumbnail", "caption", "clip_score", "gt", "sampler_id"]
            data = []

            image_captions = results['captions']
            for j,i in enumerate(image_captions):
                image_id = i['image_id']
                thumbnail = self.validator.load_image_by_id(image_id)
                thumbnail.thumbnail(thumbnail_size)
                image_url = i['image_url']
                for sr in i['sampling_results']:
                    sampler_id = sr['sampler_id']
                    for c in sr['captions']:
                        data.append([
                            image_id,
                            image_url,
                            wandb.Image(thumbnail),
                            c['caption'],
                            c['clip_score'],
                            c['gt'],
                            sampler_id
                        ])
                if j+1 >= max_log_samples:
                    break

            self.logger.log_table(key="caption_samples", columns=columns, data=data)
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
            
        embedding_text = self.language_model.get_embedding_text(tokens)

        prefix_projections = self.clip_project(prefix)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        device = tokens.device
        mask = torch.cat((torch.ones(prefix_projections.shape[:-1], dtype=torch.bool, device=device), mask), dim=1)  # adding prefix mask

        if labels is not None:
            dummy_token = torch.zeros(tokens.shape[0], self.hparams.prefix_length, dtype=torch.int64)
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        out = self.language_model.call(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        return out
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """ Callback to calculate and log grad_norm. """
        grad_norm = self.autoclip.compute_grad_norm(self)
        self.log("train/grad_norm", grad_norm)
    
    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        grad_norm = self.autoclip.compute_grad_norm(self)
        self.log("train/grad_norm_pre_clip", grad_norm)
        
        if gradient_clip_val < 0:
            self.autoclip(self)
        else:
            super().configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)

    def configure_optimizers(self):
        """ Returns a dict containing the model's optimizer and loss rate scheduler. """

        if self.hparams.use_deepspeed:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(self.parameters(), lr=self.hparams.optimizer_lr, adam_w_mode=True)
        else: 
            from torch.optim import AdamW
            optimizer = AdamW(self.parameters(), lr=self.hparams.optimizer_lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.scheduler_warmup_steps,
            num_training_steps=self.hparams.total_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        The model's main training step.
        `batch` contains a tuple of the caption's tokens, attention mask and the CLIP embedding (prefix). [see `dataset.py`]
        """

        image_tensor = batch["image_tensor"]
        tokens = batch["tokens"]

        prefix = self.encode_image(image_tensor)

        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0

        outputs = self(tokens, prefix, mask)

        logits = outputs.logits[:, self.hparams.prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        self.log_dict({
            "train/loss": loss.float(),
            "train/current_epoch": self.current_epoch,
        })
        
        return loss


class CLIPCaptionPrefixOnly(CLIPCaptionModel):
    def parameters(self):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self