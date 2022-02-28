from typing import Dict, Union, Tuple, List, Optional
import inspect
import argparse
import json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import clip
from create_dataset import CocoImageDataset
from model import CLIPCaptionModel, CLIPCaptionPrefixOnly, CaptionValidator
from lms import (
    GPT2, GPT2_Tokenizer,
    GPTJ, GPTJ_Tokenizer,
    T0, T0_Tokenizer
)

from pycocoevalcap.eval import Bleu, Cider
from pycocoevalcap.eval import PTBTokenizer


def generate_scores(gts, res):
    tokenizer = PTBTokenizer()

    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]

    output = {}
    img_output = {}

    for scorer, method in scorers:
        #print('computing {} score...'.format(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) != list:
            method = [method]
            score = [score]
            scores = [scores]

        for sc, scs, m in zip(score, scores, method):
            #print("%s: %0.3f" % (m, sc))
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
    embeds: torch.Tensor,
    top_p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    text_prefix_tokens: Optional[torch.Tensor] = None,
    max_decode_length: int = 75,
    temperature: float = 1.0,
    stop_token: str = '.',
    repetition_penalty: float = 1.2,
    max_stops = 3
):
    assert max_decode_length <= 77, "maximum context length for CLIP models is 77"

    tokenizer = model.tokenizer
    special_tokens = tokenizer.all_special_ids
    stop_token = tokenizer.encode_text(stop_token)[0]
    generations = []

    # ## BOS test
    # prepend BOS token
    bos = torch.empty(embeds.shape[0], 1, device=embeds.device, dtype=torch.int64).fill_(tokenizer.bos_token_id)
    if text_prefix_tokens is None:
        text_prefix_tokens = bos
    else:
        text_prefix_tokens = torch.cat((bos, text_prefix_tokens), dim=1)

    with torch.no_grad():
        if text_prefix_tokens is not None:
            text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
            embeds = torch.cat((embeds, text_prefix_embed), dim=1)

        embeds_init = embeds
        for top_p in top_p_values:
            tokens = None
            embeds = embeds_init
            stops = 0
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
                    stops += 1
                if stops >= max_stops or next_token.item() in special_tokens: 
                    break

            output_list = list(x for x in tokens.squeeze(0).cpu().numpy() if x not in special_tokens)
            output_text = tokenizer.decode_tokens(output_list)

            generations.append(output_text)

    return generations


@torch.no_grad()
def generate_clip_guided(
    device,
    clip_image_embedding: torch.Tensor,
    model: Union[CLIPCaptionModel, CLIPCaptionPrefixOnly],
    clip_model,
    embeds: torch.Tensor,
    text_prefix_tokens: Optional[torch.Tensor] = None,
    max_decode_length: int = 75,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    look_ahead = 5,
    branching_factor = 3,
    step_by_step = False
):
    assert max_decode_length <= 77, "maximum context length for CLIP models is 77"

    if clip_image_embedding.dim() == 3 and clip_image_embedding.shape[-2] > 1:
         clip_image_embedding = clip_image_embedding[:,0,:]

    tokenizer = model.tokenizer
    special_tokens = tokenizer.all_special_ids

    greedy = True
    top_p = 0.1

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

        # Get the next token and its embedding
        if greedy:
            val,idx = logits.topk(branching_factor)
        else:
            # Apply temperature and filter
            logits = logits / (temperature if temperature > 0 else 1.0)
            #logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=0.0)

            probabilities = F.softmax(logits, dim=-1)
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

            stop = next_token.item() in special_tokens
            if remaining_depth == 0 or stop:
                candidates.append((next_tokens, next_embeds, stop))
            else:
                recursive_branching_topk(candidates, next_embeds, next_tokens, branching_factor, remaining_depth-1)

    # prepend BOS token
    # ## BOS test 
    bos = torch.empty(embeds.shape[0], 1, device=embeds.device, dtype=torch.int64).fill_(tokenizer.bos_token_id)
    if text_prefix_tokens is None:
        text_prefix_tokens = bos
    else:
        text_prefix_tokens = torch.cat((bos, text_prefix_tokens), dim=1)

    if text_prefix_tokens is not None:
        text_prefix_embed = model.language_model.get_embedding_text(text_prefix_tokens)
        embeds = torch.cat((embeds, text_prefix_embed), dim=1)

    tokens = None

    while True:
        candidates = []
        
        current_length = tokens.shape[-1] if tokens is not None else 0
        recursive_branching_topk(
            candidates,
            embeds=embeds.clone(),
            tokens=tokens.clone() if tokens is not None else None,
            branching_factor=branching_factor,
            remaining_depth=min(look_ahead, max_decode_length-current_length)
        )

        candidate_texts = [tokenizer.decode_tokens(c[0].squeeze().tolist()) for c in candidates]

        # encode all candidate texts with clip
        candidate_clip_tokens = clip.tokenize(candidate_texts, truncate=True).to(device)
        candidate_clip_embeddings = clip_model.encode_text(candidate_clip_tokens).float()

        # cosine similarity
        clip_image_embedding = clip_image_embedding / torch.norm(clip_image_embedding)
        candidate_clip_embeddings = candidate_clip_embeddings / torch.norm(candidate_clip_embeddings, dim=-1, keepdim=True)
        similarities = clip_image_embedding @ candidate_clip_embeddings.T

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
            if tokens.shape[-1] >= max_decode_length or tokens[0, -1].item() == stop_token or tokens[0, -1].item() in special_tokens:
                break
        else:
            tokens,embeds = best_tokens,best_embeds
            if stop or tokens.shape[-1] >= max_decode_length:
                break

    output_list = list(x for x in tokens.squeeze().cpu().numpy() if x not in special_tokens)
    output_text = tokenizer.decode_tokens(output_list)
    #print(output_list, output_text)
    return output_text


def cosine_similarity(a, b):
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return a @ b.T


class ClipScoring:
    def __init__(self, clip_model, clip_image_preprocess):
        self.clip_model = clip_model
        self.clip_image_preprocess = clip_image_preprocess

    def score_image(self, image, caption, method='cosine_similarity'):
        image_tensor = self.preprocess_image(image)
        caption_tokens = self.tokenize(caption)
        return self.score_tensor(image_tensor, caption_tokens, method)

    def embed_image(self, image_tensor):
        device = next(self.clip_model.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        image_embedding = self.clip_model.encode_image(image_tensor).float()
        return image_embedding

    @torch.no_grad()
    def score_tensor(self, image_tensor, caption_tokens, method='cosine_similarity'):
        image_embedding = self.embed_image(image_tensor)
        caption_tokens = caption_tokens.to(image_embedding.device)
        text_embedding = self.clip_model.encode_text(caption_tokens).float()
        if method == 'cosine_similarity':
            score = cosine_similarity(image_embedding, text_embedding)
        else:
            raise ValueError(f'Invalid value for parameter method: {method}')
        return score

    def preprocess_image(self, image):
        return self.clip_image_preprocess(image)

    def tokenize(self, text):
        return clip.tokenize(text, truncate=True)


class CaptionSamplerBase:
    def sample(self, model, image_tensor, image):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_embedding = model.visual_encoder(image_tensor)
        prefix = model.clip_project(image_embedding)
        return self.generate_captions(model, prefix, image_embedding, image)

    def get_description(self):
        raise NotImplementedError()

    def generate_captions(self, model, prefix, image_embedding, image):
        raise NotImplementedError()


class NoBeamCaptionSampler(CaptionSamplerBase):
    def __init__(self,
        top_p_values=[0.1],
        temperature: float = 1.0,
        repetition_penalty: float = 1.2
    ):
        self.top_p_values = top_p_values
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def get_description(self):
        return f'NoBeam(rep_p={self.repetition_penalty}, temp={self.temperature}, top_p={self.top_p_values})'

    @torch.no_grad()
    def generate_captions(self, model, prefix, image_embedding, image):
        return generate_no_beam(model, prefix, top_p_values=self.top_p_values, temperature=self.temperature, repetition_penalty=self.repetition_penalty)


class ClipGuidedCaptionSampler(CaptionSamplerBase):
    def __init__(self,
        clip_scoring: ClipScoring,
        branching_factor: int=3,
        look_ahead: int=4,
        repetition_penalty: float=1.2
    ):
        self.clip_scoring = clip_scoring
        self.branching_factor = branching_factor
        self.look_ahead = look_ahead
        self.repetition_penalty = repetition_penalty

    def get_description(self):
        return f'ClipGuided(branching={self.branching_factor}, look_ahead={self.look_ahead}, rep_p={self.repetition_penalty})'

    @torch.no_grad()
    def generate_captions(self, model, prefix, image_embedding, image):
        device = next(model.parameters()).device

        clip_image_tensor = self.clip_scoring.preprocess_image(image)
        clip_image_embedding = self.clip_scoring.embed_image(clip_image_tensor)

        caption = generate_clip_guided(
            device,
            clip_image_embedding,
            model,
            self.clip_scoring.clip_model,
            prefix,
            branching_factor=self.branching_factor,
            look_ahead=self.look_ahead
        )
        return [caption]


class CocoCaptionValidator(CaptionValidator):
    def __init__(self, dataset: CocoImageDataset, preprocess, caption_samplers: Dict[str, CaptionSamplerBase], clip_scoring: ClipScoring):
        self.dataset = dataset
        self.preprocess = preprocess
        self.caption_samplers = caption_samplers
        self.clip_scoring = clip_scoring
        self.gt_captions_by_image_id = dataset.get_index().get_captions_by_image_id()
        self.reset()

    def reset(self):
        self.ground_truth_captions = {}
        self.caption_hypo = {}
        for sampler_id in self.caption_samplers.keys():
            self.caption_hypo[sampler_id] = {}
        self.results = {
            'captions': []
        }
        self.losses = []
        self.clip_scores = []

    @torch.no_grad()
    def process(self, model, batch):
        device = next(model.parameters()).device
        batch = [x for x in batch if x is not None]

        image_tensors = []
        image_captions_gt = []

        ground_truth_captions = self.ground_truth_captions
        caption_hypo = self.caption_hypo
        image_results = self.results['captions']

        for item in batch:
            image_entry = item['image_entry']
            image = item['image']

            gt = self.gt_captions_by_image_id[image_entry.id]
            ground_truth_captions[image_entry.id] = [{'caption': caption} for caption in gt]
            image_captions_gt.append(gt)

            image_tensor = self.preprocess(image).to(device)
            image_tensors.append(image_tensor)

            sampling_results = []
            for sampler_id,sampler in self.caption_samplers.items():
                captions = sampler.sample(model, image_tensor, image)

                caption = captions[0]
                caption_hypo[sampler_id][image_entry.id] = [{'caption': caption}]

                # compute clip score
                clip_scores = self.clip_scoring.score_image(image, captions)

                captions_result = []
                for i,c in enumerate(captions):
                    cs = clip_scores[0, i].item()
                    captions_result.append(
                        { 'caption': captions[i], 'clip_score': cs, 'gt': gt[0] }
                    )
                    self.clip_scores.append(cs)

                sampling_results.append(
                    {
                        'sampler_id': sampler_id,
                        'captions': captions_result
                    }
                )

            image_result = {
                'image_id': image_entry.id,
                'image_url': image_entry.url,
                'sampling_results': sampling_results
            }
            image_results.append(image_result)

        # evaluate loss of model
        image_batch = torch.stack(image_tensors, dim=0)
        prefixes = model.visual_encoder(image_batch)

        min_cap_per_img = min(len(x) for x in image_captions_gt)    # all entries in batch have >= min_cap_per_img captions 
        for i in range(min_cap_per_img):
            encoded_text = [torch.tensor(model.tokenizer.encode_text(c[i], add_bos=True, add_eos=True), dtype=torch.int64) for c in image_captions_gt]
            max_len = max(s.shape[-1] for s in encoded_text)
            tokens = torch.zeros(len(encoded_text), max_len, dtype=torch.int64)
            for i,t in enumerate(encoded_text):
                tokens[i, :t.shape[-1]] = t

            tokens = tokens.to(device)
            mask = tokens.ge(0)  # mask is zero where we out of sequence
            tokens[~mask] = 0

            outputs = model.forward(tokens, prefixes, mask)
            logits = outputs.logits[:, model.hparams.prefix_length - 1: -1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            self.losses.append(loss.item())


    def get_results(self):
        results = self.results

        # Calculate scores
        sampler_scores = {}
        for sample_id, sampler_hypo in self.caption_hypo.items():
            scores, img_scores = generate_scores(self.ground_truth_captions,  sampler_hypo)
            sampler_scores[sample_id] = scores

        results['validation_loss'] = np.mean(self.losses)
        results['clip_score'] = np.mean(self.clip_scores)
        results['sampler_scores'] = sampler_scores

        return results

    def load_image_by_id(self, image_id):
        return self.dataset.load_image_by_id(image_id)


def evaluate(
    device: torch.device,
    checkpoint_path: str,
    clip_model: str,
    language_model_type: str,
    language_model_variant: str,
    prefix_only: bool,
    valid_json_path: str,
    image_folder_path: str,
    visual_encoder_type: str='BLIP',
    visual_encoder_model_variant: str='ViT-B',
    hf_cache_dir: Optional[str] = None
):

    print(f"loading CLIP: '{clip_model}'")
    clip_model, clip_image_preprocess = clip.load(clip_model, device=device, jit=False)

    if visual_encoder_type == 'BLIP':
        if visual_encoder_model_variant == 'ViT-B':
            blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
        else:
            raise RuntimeError('Visual encoder model variant not supported: \'{visual_encoder_model_variant}\'')

        image_size = 384
        preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        from BLIP.models.blip import blip_decoder

        blip_model = blip_decoder(pretrained=blip_model_url, image_size=image_size, vit='base', med_config='BLIP/configs/med_config.json')
        blip_model.eval()

        blip_model = blip_model.to(device)
        visual_encoder = blip_model.visual_encoder
    else:
        raise RuntimeError('Unsupported visual encdore \'{visual_encoder_type}\' specified.')

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

    dataset = CocoImageDataset(annotation_json_path=valid_json_path, image_folder_path=image_folder_path)
    gt_captions_by_image_id = dataset.get_index().get_captions_by_image_id()

    if prefix_only:
        model = CLIPCaptionPrefixOnly.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, tokenizer=tokenizer, validator=None, visual_encoder=visual_encoder, strict=False)
    else:
        model = CLIPCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, language_model=language_model, tokenizer=tokenizer, visual_encoder=visual_encoder, validator=None, strict=False)

    model.to(device)
    model.eval()

    nobeam_sampler = NoBeamCaptionSampler(top_p_values=[0.1, 0.2])
    clip_scoring = ClipScoring(clip_model, clip_image_preprocess)
    clip_guided_sampler = ClipGuidedCaptionSampler(clip_scoring, branching_factor=2, look_ahead=4)
    validator = CocoCaptionValidator(
        dataset,
        preprocess,
        {
            'nobeam': nobeam_sampler,
            'clip_guided': clip_guided_sampler
        },
        clip_scoring
    )

    max_samples = 10
    for i,x in enumerate(tqdm(dataset, desc='inference')):
        dummy_batch = [x]
        validator.process(model, dummy_batch)
        if i+1 >= max_samples:
            break

    results = validator.get_results()
    print(results)


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
    parser.add_argument('--clip_model', default= "ViT-B/32", type=str, help="available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']")
    parser.add_argument('--language_model_type', default="gpt2", type=str)
    parser.add_argument('--language_model_variant', default="gpt2", type=str, help='gpt2, gpt2-xl')
    parser.add_argument('--prefix_only', default=False, type=str2bool)
    parser.add_argument('--hf_cache_dir', default=None, type=str)

    parser.add_argument('--valid_json_path', default='/data/datasets/coco/annotations/captions_val2017.json', type=str)
    parser.add_argument('--image_folder_path', default='/data/datasets/coco/val2017/', type=str)

    parser.add_argument('--visual_encoder_type', type=str, default='BLIP')
    parser.add_argument('--visual_encoder_model_variant', type=str, default='ViT-B')

    parser.add_argument('--checkpoint_path', type=str, default='./out/006_0.ckpt')

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

    evaluate(**merge_args(evaluate, vars(opt), device=device))


if __name__ == '__main__':
    main()
