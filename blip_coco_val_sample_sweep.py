import math
import argparse
from pathlib import Path
from PIL import Image
import random
from typing import Dict
import numpy as np
import torch

import clip
from create_dataset import CocoJsonDataset
from evaluate_model import generate_scores
from sampling import blip_rank, clip_rank, load_blip_decoder, load_blip_ranking_model, sample



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--valid_json_path', default='/data/datasets/coco/annotations/captions_val2017.json', type=str)
    parser.add_argument('--image_folder_path', default='/data/datasets/coco/val2017/', type=str)
    parser.add_argument('--n', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_sampling_runs', default=1, type=int)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--torch_hub_dir', default='/mnt/sdb3/torch_hub', type=str)
    parser.add_argument('--mode', default='ITC', type=str)   # CLIP-ViT-L+RN50x64, CLIP-ViT-L, CLIP-RN50x64, ITC, ITM
  
    parser.add_argument('--deviceA_index', default=0, type=int)
    parser.add_argument('--deviceB_index', default=1, type=int)

    opt = parser.parse_args()
    return opt


class RankingModel:
    def __init__(self, mode, device0, device1):
        self.mode = mode
        self.device0 = device0
        self.device1 = device1
        
        if mode == 'CLIP-ViT-L+RN50x64':
            clip_model_name1 = "ViT-L/14"
            clip_model_name2 = "RN50x64"
            print('loading CLIP: ', clip_model_name1)
            self.clip_model1, self.clip_preprocess1 = clip.load(clip_model_name1, device=device1)
            print('loading CLIP: ', clip_model_name2)
            self.clip_model2, self.clip_preprocess2 = clip.load(clip_model_name2, device=device0)
        elif mode == 'CLIP-ViT-L':
            clip_model_name1 = "ViT-L/14"
            print('loading CLIP: ', clip_model_name1)
            self.clip_model1, self.clip_preprocess1 = clip.load(clip_model_name1, device=device1)
        elif mode == 'CLIP-RN50x64':
            clip_model_name1 = "RN50x64"
            print('loading CLIP: ', clip_model_name1)
            self.clip_model1, self.clip_preprocess1 = clip.load(clip_model_name1, device=device1)
        elif mode == 'ITC' or mode == 'ITM':
            self.blip_ranking_model = load_blip_ranking_model(device0)
        else:
            raise RuntimeError(f'Unsupported mode "{mode}"')

    def rank(self, raw_image, captions):
        if self.mode == 'CLIP-ViT-L+RN50x64':
            sims = clip_rank(self.device1, self.clip_model1, self.clip_preprocess1, raw_image, captions)
            top_indices = np.argsort(np.asarray(sims))[-5:][::-1]
            best_captions = [captions[i] for i in top_indices]
            sims2 = clip_rank(self.device0, self.clip_model2, self.clip_preprocess2, raw_image, best_captions)
            best_index = np.argmax(np.asarray(sims2))
            synth_caption = best_captions[best_index]
        elif self.mode == 'CLIP-ViT-L' or self.mode == 'CLIP-RN50x64':
            sims = clip_rank(self.device1, self.clip_model1, self.clip_preprocess1, raw_image, captions)
            best_index = np.argmax(np.asarray(sims))
            synth_caption = captions[best_index]
        elif self.mode == 'ITC' or self.mode == 'ITM':
            sims = blip_rank(self.device0, self.blip_ranking_model, raw_image, captions, mode=self.mode.lower())
            best_index = np.argmax(np.asarray(sims))
            synth_caption = captions[best_index]
        
        return synth_caption


def main():
    args = parse_args()

    seed = args.manual_seed
    torch.manual_seed(seed)
    random.seed(seed)

    n = args.n

    val_annotations = CocoJsonDataset(args.valid_json_path)
    image_ids = list(val_annotations.image_by_id.keys())
    image_ids.sort()

    # select validation images to generate captions for
    image_ids = [image_ids[x] for x in torch.randperm(len(image_ids))[:n]]
    captions_by_image_id = val_annotations.get_captions_by_image_id()
    
    ground_truth_captions = {}
    for id in image_ids:
        gt = captions_by_image_id[id]
        ground_truth_captions[id] = [{'caption': caption} for caption in gt]

    torch.hub.set_dir(args.torch_hub_dir)

    device0 = torch.device('cuda', args.deviceA_index)
    device1 = torch.device('cuda', args.deviceB_index)

    model,transform = load_blip_decoder(device1)
    ranking = RankingModel(args.mode, device0, device1)

    min_len = 5
    max_len = 45
    top_k = 2500
    top_p = 0.5
    force_eos_prob = 1.0
    batch_size = 40

    top_p = torch.tensor([top_p] * batch_size, device=device1)
    min_len = torch.tensor(([min_len] * batch_size), device=device1)
    max_len = torch.tensor(([max_len] * batch_size), device=device1)


    image_folder_path = Path(args.image_folder_path)
    for i,id in enumerate(image_ids):
        image = val_annotations.image_by_id[id]
        f = image_folder_path / image.file_name

        raw_image = Image.open(f).convert('RGB')

        image = transform(raw_image).unsqueeze(0).to(device1)

        captions,_,_ = sample(
            image,
            model,
            sample_count=min_len.size(0),
            top_p=top_p,
            top_k=top_k,
            min_len=min_len,
            max_len=max_len,
            force_eos_log_prob=math.log(force_eos_prob),
            prompt='a picture of ',
            num_runs=args.num_sampling_runs)

        synth_caption = ranking.rank(raw_image, captions)
        print(synth_caption)



if __name__ == '__main__':
    main()
