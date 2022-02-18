from typing import Union, Tuple, List, Optional
import inspect
import argparse
import json
import glob
from tqdm import tqdm


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image

import clip
from clip.model import VisionTransformer
from create_dataset import CocoImageDataset
from model import CLIPCaptionModel, CLIPCaptionPrefixOnly
from lms import (
    GPT2, GPT2_Tokenizer,
    GPTJ, GPTJ_Tokenizer,
    T0, T0_Tokenizer
)

from pycocoevalcap.eval import Bleu
from pycocoevalcap.eval import PTBTokenizer


def generate_scores(gts, res):
    tokenizer = PTBTokenizer()

    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        # (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]

    output = {}
    img_output = {}

    for scorer, method in scorers:
        print('computing {} score...'.format(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) != list:
            method = [method]
            score = [score]
            scores = [scores]

        for sc, scs, m in zip(score, scores, method):
            print("%s: %0.3f" % (m, sc))
            output[m] = sc
            for img_id, score in zip(gts.keys(), scs):
                if type(score) is dict:
                    score = score['All']['f']

                if img_id not in img_output:
                    img_output[img_id] = {}
                img_output[img_id][m] = score

    return output, img_output


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
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

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


def generate_no_beam(
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    top_p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    text_prefix_tokens: Optional[torch.Tensor] = None,
    max_decode_length: int = 96,
    temperature: float = 1.0,
    stop_token: str = '.',
    repetition_penalty: float = 1.2,
):

    stop_token = tokenizer.encode_text(stop_token)[0]
    generations = []

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for top_p in top_p_values:
            tokens = None
            embeds = embeds_init
            for _ in range(max_decode_length):
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
                probabilities = F.softmax(logits, dim=-1)
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


@torch.no_grad()
def generate_hill(
    device,
    clip_embedding,
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    clip_model,
    tokenizer: GPT2_Tokenizer,
    embeds: torch.Tensor,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    max_decode_length: int = 70,
    temperature: float = 1.0,
    stop_token: str = '.',
    repetition_penalty: float = 1.2,
    look_ahead = 4,
    branching_factor = 3,
    step_by_step = False
):

    if clip_embedding.dim() == 3 and clip_embedding.shape[-2] > 1:
         clip_embedding = clip_embedding[:,0,:]
    print('clip_embedding', clip_embedding.shape)

    stop_token = tokenizer.encode_text(stop_token)[0]
    generations = []

    greedy = True
    top_p = 0.1
    temperature = 1.0

    def recursive_branching_topk(candidates, embeds, tokens, branching_factor=3, remaining_depth=4):
        assert embeds.shape[0] == 1 # Assume batch size of 1
        assert remaining_depth >= 0 and branching_factor > 0

        # Get logits from a forward pass
        outputs = model.language_model.call(inputs_embeds=embeds)
        logits = outputs.logits[0, -1, :]

        # Apply the repetition penalty
        if repetition_penalty != 1.0 and tokens is not None:
            tokens1 = tokens[0, :] # assuming batch size of 1
            logits = repetition_penalty_apply(logits, tokens1, repetition_penalty)

        # Apply temperature and filter
        logits = logits / (temperature if temperature > 0 else 1.0)
        #logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=0.0)
 
        # Get the next token and its embedding
        probabilities = F.softmax(logits, dim=-1)
        if greedy:
            val,idx = probabilities.topk(branching_factor)
        else:
            idx = torch.multinomial(probabilities, branching_factor, replacement=False)

        #print(val, idx)
        for next_token in idx:
            next_token = next_token.view(1, -1)
            next_token_embed = model.language_model.get_embedding_text(next_token)
            
            if tokens is None:
                next_tokens = next_token
            else:
                next_tokens = torch.cat((tokens, next_token), dim=1)
            #print('embeds', embeds.shape)
            #print('next_token_embed', next_token_embed.shape)
            next_embeds = torch.cat((embeds, next_token_embed), dim=1)

            stop = next_token.item() == stop_token
            if remaining_depth == 0 or stop:
                candidates.append((next_tokens, next_embeds, stop))
            else:
                recursive_branching_topk(candidates, next_embeds, next_tokens, branching_factor, remaining_depth-1)


    if text_prefix_tokens is not None:
        text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
        embeds = torch.cat((embeds, text_prefix_embed), dim=1)

    tokens = None

    for j in range(10):
        candidates = []

        recursive_branching_topk(candidates, embeds=embeds.clone(), tokens=tokens.clone() if tokens is not None else None, branching_factor=branching_factor, remaining_depth=look_ahead)
        candidate_texts = [tokenizer.decode_tokens(c[0].squeeze().tolist()) for c in candidates]

        # encode all candidate texts with clip
        candidate_clip_tokens = clip.tokenize(candidate_texts).to(device)
        candidate_clip_embeddings = clip_model.encode_text(candidate_clip_tokens).float()

        # cosine similarity
        clip_embedding = clip_embedding / torch.norm(clip_embedding)
        candidate_clip_embeddings = candidate_clip_embeddings / torch.norm(candidate_clip_embeddings, dim=-1, keepdim=True)
        similarities = clip_embedding @ candidate_clip_embeddings.T

        best = similarities.argmax()
        #print('best', candidate_texts[best])

        best_tokens, best_embeds, stop = candidates[best]

        if step_by_step:
            if tokens is None:
                tokens = best_tokens[:, :1]
            else:
                tokens = best_tokens[:, :tokens.shape[-1]+1]
            #print('before:', best_tokens)
            embeds = best_embeds[:, :embeds.shape[1]+1, :]
            #print('after', tokens, stop_token)
            if tokens[0, -1].item() == stop_token:
                break
        else:
            tokens,embeds = best_tokens,best_embeds
            if stop:
                break

    output_list = list(tokens.squeeze().cpu().numpy())
    output_text = tokenizer.decode_tokens(output_list)
    #print(output_list, output_text)
    return output_text


def generate_caption(device, language_model, tokenizer, clip_model, image_tensor, top_p_values=[0.1]):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        prefix = clip_model.encode_image(image_tensor).float()
        prefix_embed = language_model.clip_project(prefix)
    #generated_captions = generate_no_beam(language_model, tokenizer, prefix_embed, top_p_values, text_prefix_tokens=None)
    generated_caption = generate_hill(device, prefix, language_model, clip_model, tokenizer, prefix_embed)
    return generated_caption


def evaluate(
    device: torch.device,
    checkpoint_path: str,
    clip_model: str, 
    use_all_vit_features: bool, 
    language_model_type: str, 
    language_model_variant: str, 
    prefix_only: bool, 
    valid_json_path: str,
    image_folder_path: str,
    hf_cache_dir: Optional[str] = None,
):
    max_token_length = 128

    print(f"loading CLIP: '{clip_model}'")
    clip_model, preprocess = clip.load(clip_model, device=device, jit=False)

    if use_all_vit_features:
        # original: https://github.com/openai/CLIP/blob/40f5484c1c74edd83cb9cf687c6ab92b28d8b656/clip/model.py#L202-L236
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
            #x = self.ln_post(x[:, 0, :])

            if self.proj is not None:
                x = x @ self.proj

            return x

        clip_model.visual.forward = vit_forward_patch.__get__(clip_model.visual, VisionTransformer)

    print(f"loading LM variant: '{language_model_variant}'")
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

    dataset = CocoImageDataset(annotation_json_path=valid_json_path, image_folder_path=image_folder_path, image_transform=preprocess)
    gt_captions_by_image_id = dataset.annotations.get_captions_by_image_id()    

    if prefix_only:
        language_model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, strict=False)
    else:
        language_model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, strict=False)

    language_model.to(device)
    language_model.eval()

    ## hack
    results = []
    files = glob.glob('/data/datasets/persons_test/*.jpg')
    dataset = FileListImageDataset(files, transform=preprocess)
    for x in tqdm(dataset, desc='inference'):
        image_tensor = x['image_tensor']
        file_name = x['file_name']
        
        caption = generate_caption(device, language_model, tokenizer, clip_model, image_tensor, top_p_values=[0.1])
        print(file_name)
        print(caption)
        results.append(
            {
                'file_name': file_name,
                'caption': caption
            }
        )
    return results
    

    ground_truth_captions = {}
    caption_hypo = {}

    for x in tqdm(dataset, desc='inference'):
        image_tensor = x['image_tensor']
        image_entry = x['image_entry']
        print('image-url: ', image_entry.url)
        caption = generate_captions(device, language_model, tokenizer, clip_model, image_tensor, top_p_values=[0.1])[0]
        caption_hypo[image_entry.id] = [{'caption': caption}]
        ground_truth_captions[image_entry.id] = [{'caption': caption} for caption in gt_captions_by_image_id[image_entry.id]]

    # Calculate scores
    scores, img_scores = generate_scores(ground_truth_captions, caption_hypo)
    print("Scores")
    print(scores)


# parse bool args correctly, see https://stackoverflow.com/a/43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--device-index', default=0, type=int, help='device index')
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--clip_model', default= "ViT-B/32", type=str, help="available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']")
    parser.add_argument('--use_all_vit_features', default=True, type=str2bool)
    parser.add_argument('--language_model_type', default="gpt2", type=str)
    parser.add_argument('--language_model_variant', default="gpt2", type=str, help='gpt2, gpt2-xl')
    parser.add_argument('--prefix_only', default=False, type=str2bool)
    parser.add_argument('--hf_cache_dir', default=None, type=str)

    parser.add_argument('--valid_json_path', default='/data/datasets/coco/annotations/captions_val2017.json', type=str)
    parser.add_argument('--image_folder_path', default='/data/datasets/coco/val2017/', type=str)

    parser.add_argument('--checkpoint_path', type=str, default='./out/002_coco2017_gpt2_po_allfeat.ckpt_epoch_4.ckpt')
    #parser.add_argument('--checkpoint_path', type=str, default='./out/003_coco2017_gpt2_po_allfeat_pos.ckpt_final.ckpt')
    parser.add_argument('--load_pl_checkpoint', default=True, type=str2bool)

    opt = parser.parse_args()
    return opt


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Command line args:', opt)

    device = torch.device(opt.device, opt.device_index)
    print('Device:', device)
    
    def merge_args(fn, args, **kwargs):
        a = {k: args[k] for k in inspect.signature(fn).parameters if k in args}
        for k,v in kwargs.items():
            a[k] = v
        return a

    results = evaluate(**merge_args(evaluate, vars(opt), device=device))

    # Save the results
    out_filename = f'persons_default.json'
    with open(out_filename, "w+") as f:
        json.dump(results, f)



class FileListImageDataset(Dataset):
    def __init__(self, file_names, transform=None):
        super(FileListImageDataset, self).__init__()
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        fn = self.file_names[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return { 'image_tensor': img, 'file_name': fn }


if __name__ == '__main__':
    main()
