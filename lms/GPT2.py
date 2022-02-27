from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List
import torch


class GPT2(GPT2LMHeadModel):
    @classmethod
    def create(cls, model_variant: str = "gpt2-xl", **huggingface_kwargs):
        return cls.from_pretrained(model_variant, **huggingface_kwargs)

    def get_embedding_size(self) -> int:
        return self.transformer.wte.weight.shape[1]
    
    def get_embedding_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(tokens)
    
    def call(self, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)


class GPT2_Tokenizer(GPT2Tokenizer):
    @classmethod
    def create(cls, model_variant: str = "gpt2-xl", **huggingface_kwargs):
        tokenizer = cls.from_pretrained(model_variant, **huggingface_kwargs)
        return tokenizer

    def encode_text(self, text: str, max_token_length: Optional[int] = None, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if max_token_length is not None:
            if add_bos:
                max_token_length += 1
            if add_eos:
                max_token_length += 1

        tokens = self.encode(text)
        if max_token_length is not None:
            tokens = tokens[:max_token_length]

        if add_bos:
            tokens = [self.bos_token_id] + tokens

        if add_eos:
            tokens = tokens + [self.eos_token_id]
        
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> str:
        return self.decode(tokens)
