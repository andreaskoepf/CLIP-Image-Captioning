from transformers import CLIPModel, CLIPProcessor
from typing import Tuple, List, Optional
import torch.nn.functional as nnf
from typing import Union
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import fire

from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import (
    GPT2, GPT2_Tokenizer,
    GPTJ, GPTJ_Tokenizer,
    T0, T0_Tokenizer
)
from utils import scoring

# From https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def repetition_penalty_apply(logits, tokens, penalty):
    tok_logits = torch.gather(logits, -1, tokens)
    tok_logits = torch.where(tok_logits < 0, tok_logits * penalty, tok_logits / penalty)
    logits.scatter_(-1, tokens, tok_logits)
    return logits

def generate_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    beam_size: int = 5,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None
    scores = None

    seq_lengths = torch.ones(beam_size, device=embeds.device)
    has_stopped = torch.zeros(beam_size, dtype=torch.bool, device=embeds.device)
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model.call(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()

                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    embeds = embeds.expand(beam_size, *embeds.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[has_stopped] = -float(np.inf)
                    logits[has_stopped, 0] = 0

                    scores_sum = scores[:, None] + logits
                    seq_lengths[~has_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)

                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)

                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)

                    embeds = embeds[next_tokens_source]
                    scores = scores_sum_average * seq_lengths

                    has_stopped = has_stopped[next_tokens_source]
                
                next_token_embed = model.language_model.get_embedding_text(next_tokens.squeeze()).view(embeds.shape[0], 1, -1)
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                has_stopped = has_stopped + next_tokens.eq(stop_token).squeeze()
                if has_stopped.all():
                    break
                
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode_tokens(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order][0]

            generations.append(output_texts)
    
    return generations


def generate_nucleus_sampling(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = '.'
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    tokens = None

    filter_value = -float(np.inf)
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        for i in range(number_to_generate):
            for _ in range(entry_length):
                outputs = model.language_model.call(inputs_embeds=embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                if top_k is None:
                    top_k = logits.shape[-1]
                if top_p is None:
                    top_p = 1.0
                    
                p, largest_p_idx = nnf.softmax(logits, dim=-1).topk(top_k, dim=-1)
                cumulative_p = p.cumsum(dim=-1)
                threshold_repeated = top_p + torch.zeros((len(p), 1)).to("cuda:4")
                idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k-1).squeeze()
                cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
                censored_p = (cumulative_p <= cutoffs[:, None]) * p
                renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

                final_p = torch.zeros_like(logits)
                row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1,top_k).to("cuda:4")
                final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

                next_token = torch.multinomial(final_p, num_samples=1)
                
                next_token_embed = model.language_model.get_embedding_text(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                
                if stop_token == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode_tokens(output_list)
        
            generations.append(output_text)
    
    return generations


def generate_no_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    number_to_generate: int = 1,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = '.',
    repetition_penalty: float = 1.2,
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    print(f'Generate_no_beam (repetition_penalty: {repetition_penalty:.2f})')
    filter_value = -float(np.inf)
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for top_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            tokens = None
            embeds = embeds_init
            for _ in range(entry_length):
                # Get logits from a forward pass
                outputs = model.language_model.call(inputs_embeds=embeds)
                logits = outputs.logits

                # Assume batch size of 1
                assert logits.shape[0] == 1
                logits = logits[0, -1, :]

                # Apply the repetition penalty
                if repetition_penalty != 1.0 and tokens is not None:
                    tokens1 = tokens[0, :] # assuming batch size of 1
                    logits = repetition_penalty_apply(logits, tokens1, repetition_penalty)

                # Apply temperature and filter
                logits = logits / (temperature if temperature > 0 else 1.0)
                logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=0.0)

                # Get the next token and its embedding
                probabilities = nnf.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).unsqueeze(0)
                next_token_embed = model.language_model.get_embedding_text(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                embeds = torch.cat((embeds, next_token_embed), dim=1)
                
                if stop_token == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode_tokens(output_list)
        
            generations.append(output_text)
    
    return generations


def demo_generate_captions(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    clip_model: CLIPModel,
    clip_preproc: CLIPProcessor,
    image: Image.Image,
    number_to_generate: int = 1,
    text_prefix: Optional[str] = None,
    use_beam_search: bool = False,
    use_all_vit_features: bool = True,
    device: str = "cuda:0",
    **generation_kwargs
) -> Tuple[List[str], torch.Tensor]:

    clip_inputs = clip_preproc(images=image, return_tensors="pt")

    with torch.no_grad():
        if use_all_vit_features:
            outputs = model.vision_model(**clip_inputs).last_hidden_state
            outputs = model.vision_model.post_layernorm(outputs)
            outputs = model.visual_projection(outputs)
        else:
            outputs = model.get_image_features(**clip_inputs)
        
        prefix_embed = model.clip_project(outputs)   #.reshape(-1, model.prefix_length, model.lm_embedding_size)
    
    if text_prefix is not None:
        text_prefix_tokens = torch.tensor(tokenizer.encode_text(text_prefix), device=device).unsqueeze(0)
    else:
        text_prefix_tokens = None
    
    if use_beam_search:
        generated_captions = generate_beam(model, tokenizer, prefix_embed,
            number_to_generate=number_to_generate, text_prefix_tokens=text_prefix_tokens,
            **generation_kwargs)
    else:
        generated_captions = generate_no_beam(model, tokenizer, prefix_embed,
            number_to_generate=number_to_generate, text_prefix_tokens=text_prefix_tokens,
            **generation_kwargs)
    
    if text_prefix is not None:
        generated_captions = [(text_prefix + caption) for caption in generated_captions]
    
    return generated_captions, (outputs[0][0] if use_all_vit_features else outputs[0])


# def demo(
#     checkpoint_path: str = "./train/latest.pt",
#     prefix_length: int = 40,
#     clip_prefix_length: int = 40,
#     prefix_size: int = 512,
#     mapping_type: str = "mlp",
#     num_layers: int = 8,
#     only_prefix: bool = False,
#     language_model_type: str = "gpt2",
#     language_model_variant: str = "gpt2-xl",
#     clip_model_type: str = "ViT-B/32",
#     load_full_model: bool = True,
#     use_beam_search: bool = False,
#     device: str = "cuda:0",
#     **generation_kwargs
# ):
#     clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

#     if language_model_type == "gpt2":
#         language_model = GPT2.create(language_model_variant)
#         tokenizer = GPT2_Tokenizer.create(language_model_variant)
#     else:
#         raise ValueError(f"invalid language model type: '{language_model_type}' (expected 'gpt2')")

#     if only_prefix:
#         if load_full_model:
#             model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path)
#         else:
#             model = CLIPCaptionPrefixOnly(
#                 language_model, prefix_length, clip_length=clip_prefix_length,
#                 prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type
#             )
#             model.load_state_dict(torch.load(checkpoint_path))
#     else:
#         if load_full_model:
#             model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
#         else:
#             model = CLIPCaptionModel(
#                 language_model, prefix_length, clip_length=clip_prefix_length,
#                 prefix_size=prefix_size, num_layers=num_layers, mapping_type=mapping_type
#             )
#             model.load_state_dict(torch.load(checkpoint_path))
    
#     model = model.to(device)
#     model = model.eval()

#     try:
#         while True:
#             print("CLIP-Image-Captioning inference demo\n")

#             image_path_url = input("enter image url or path > ")

#             image = io.imread(image_path_url)
#             image = Image.fromarray(image)

#             caption = demo_generate_captions(
#                 model, tokenizer, clip_model, preprocess, image,
#                 use_beam_search=use_beam_search, device=device, **generation_kwargs
#             )

#             print(caption)
#             print()
#     except KeyboardInterrupt:
#         print("exiting...")
#         exit(0)


def _shutterstock_demo(
    checkpoint_path: str,
    shutterstock_path: str,
    number_to_generate: int = 1,
    text_prefix: Optional[str] = None,
    device: str = "cuda:0",
    use_beam_search: bool = True,
    prefix_only: bool = False,
    out_filename_prefix: str = "demo_inference",
    hf_clip_model: str = "openai/clip-vit-large-patch14",
    language_model_type: str = "gpt2",
    language_model_variant: str = "gpt2-xl",
    hf_cache_dir: Optional[str] = None,
    total_samples: int = 100,
    load_pl_checkpoint: bool = True,
    use_all_vit_features: bool = True,
    **model_kwargs
):
    clip_model = CLIPModel.from_pretrained(hf_clip_model).eval().to(device)
    clip_preprocess = CLIPProcessor.from_pretrained(hf_clip_model)

    if language_model_type == "gpt2":
        language_model = GPT2.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = GPT2_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    elif language_model_type in ("gptj", "gpt-j"):
        language_model = GPTJ.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = GPTJ_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    elif language_model_type in ("t0", "t5"):
        language_model = T0.create(language_model_variant, cache_dir=hf_cache_dir)
        tokenizer = T0_Tokenizer.create(language_model_variant, cache_dir=hf_cache_dir)
    else:
        raise ValueError(f"invalid language model type '{language_model_type}' (expected 'gpt-j' / 'gpt2' / 't0' / 't5')")

    if load_pl_checkpoint:
        if prefix_only:
            model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, strict=False)
        else:
            model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, strict=False)
    else:
        if prefix_only:
            model = CLIPCaptionPrefixOnly(language_model, **model_kwargs)
        else:
            model = CLIPCaptionModel(language_model, **model_kwargs)
        
        model.load_state_dict(torch.load(checkpoint_path))
    
    model = model.to(device)
    model = model.eval()

    from pathlib import Path
    from tqdm import tqdm
    import json

    samples_path = Path(shutterstock_path)
    sample_data = {}

    scoring_gts = {}
    scoring_res = {}
    image_id = 0
    image_id_to_url = {}

    for image_file in tqdm(sorted(list(samples_path.glob("*.jpg")), key=lambda x: x.name)[:total_samples], desc='inference'):
        image = io.imread(image_file)
        pil_image = Image.fromarray(image)

        metadata_file = image_file.parent / image_file.name.replace(".jpg", ".json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        captions, image_features = demo_generate_captions(
            model, tokenizer, clip_model, clip_preprocess, pil_image,
            use_beam_search=use_beam_search, device=device,
            number_to_generate=number_to_generate, text_prefix=text_prefix,
            use_all_vit_features=use_all_vit_features
        )

        print(image_file)
        print(captions)

        url = metadata["src"]
        original_caption = metadata["alt"]

        text_inputs = clip_preprocess(text=[original_caption, *captions], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        #similarities = image_features.cpu().numpy() @ text_features.cpu().numpy().T
        #similarities = similarities.tolist()

        #original_sim = similarities[0][0]
        #generated_sims = similarities[0][1:]

        #best_sim = max(generated_sims)
        #best_caption = captions[generated_sims.index(best_sim)]

        sample_data[url] = {
            "original_caption": original_caption,
            #"original_sim": original_sim,
            "generated_captions": captions,
            #"generated_sim": generated_sims,
            #"best_caption": best_caption,
            #"best_sim": best_sim
        }

        # Collect the records we'll need for scoring
        scoring_res[image_id] = [
            {u'caption': original_caption}
        ]
        scoring_gts[image_id] = [
            {u'caption': caption} for caption in captions
        ]

        image_id_to_url[image_id] = url
        image_id += 1
    
    # Calculate scores
    scores, img_scores = scoring.generate_scores(scoring_gts, scoring_res)
    print("Scores")
    print(scores)

    # Integrate the scores into the results
    for img_id in img_scores.keys():
        sample_data[image_id_to_url[img_id]]["scores"] = img_scores[img_id]

    # Save the results
    with open(f"{out_filename_prefix}_shutterstock.json", "w+") as f:
        json.dump(sample_data, f)


if __name__ == "__main__":
    fire.Fire(_shutterstock_demo)
