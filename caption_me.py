import math
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sampling import clip_rank, load_blip_decoder, sample, blip_rank, load_blip_ranking_model
import clip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--print_top_n', default=5, type=int)
    
    parser.add_argument('--top_k', default=2500, type=int)
    parser.add_argument('--typ_p', default=0, type=float)
    parser.add_argument('--force_eos_prob', default=0.9, type=float)
    parser.add_argument('--num_sampling_runs', default=1, type=int)

    parser.add_argument('--mode', default='CLIP-ViT-L+RN50x64', type=str)   # CLIP-ViT-L+RN50x64, CLIP-ViT-L, CLIP-RN50x64, ITC

    parser.add_argument('--deviceA_index', default=0, type=int)
    parser.add_argument('--deviceB_index', default=1, type=int)

    parser.add_argument('--set_max_len', default=None, type=int)
    parser.add_argument('--set_min_len', default=None, type=int)
    parser.add_argument('--set_top_p', default=None, type=float)
    parser.add_argument('--torch_hub', default=None, type=str)

    opt = parser.parse_args()
    return opt


def main():
    args = parse_args()

    seed = args.manual_seed
    torch.manual_seed(seed)

    random.seed(seed)

    if args.torch_hub is not None:
        torch.hub.set_dir(args.torch_hub)   # --torch_hub /media/koepf/data2/torch_hub

    device0 = torch.device('cuda', args.deviceA_index)
    device1 = torch.device('cuda', args.deviceB_index)

    model,transform = load_blip_decoder(device1)

    mode = args.mode

    if mode == 'CLIP-ViT-L+RN50x64':
        clip_model_name1 = "ViT-L/14"
        clip_model_name2 = "RN50x64"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device1)
        print('loading CLIP: ', clip_model_name2)
        clip_model2, clip_preprocess2 = clip.load(clip_model_name2, device=device0)
    elif mode == 'CLIP-ViT-L':
        clip_model_name1 = "ViT-L/14"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device1)
    elif mode == 'CLIP-RN50x64':
        clip_model_name1 = "RN50x64"
        print('loading CLIP: ', clip_model_name1)
        clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device1)
    elif mode == 'ITC' or mode == 'ITM':
        blip_ranking_model = load_blip_ranking_model(device0)
    else:
        raise RuntimeError(f'Unsupported mode "{mode}"')

    
    raw_image = Image.open(args.input_image).convert('RGB')   
    w,h = raw_image.size

    image = transform(raw_image).unsqueeze(0).to(device1)
        
    if args.set_top_p is not None:
        top_p = torch.tensor([args.set_top_p]*40, device=device1)
    else:
        top_p = torch.tensor(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]*5), device=device1)
    
    if args.set_min_len is not None:
        min_len = torch.tensor(([args.set_min_len]*40), device=device1)
    else:
        min_len = torch.tensor(([5]*8 + [10]*8 + [15]*8 + [20]*8 + [30]*8), device=device1)
    
    if args.set_max_len is not None:
        max_len = torch.tensor(([args.set_max_len]*40), device=device1)
    else:
        max_len = torch.tensor(([20]*8 + [30]*8 + [30]*8 + [45]*8 + [45]*8), device=device1)

    typ_p = args.typ_p
    top_k = args.top_k

    captions,_,_ = sample(
            image,
            model,
            sample_count=min_len.size(0),
            top_p=top_p,
            top_k=top_k,
            typ_p=typ_p,
            min_len=min_len,
            max_len=max_len,
            force_eos_log_prob=math.log(args.force_eos_prob),
            prompt='a picture of ',
            num_runs=args.num_sampling_runs)

    # if mode == 'CLIP-ViT-L+RN50x64':
    #     sims = clip_rank(device1, clip_model1, clip_preprocess1, raw_image, captions)
    #     top_indices = np.argsort(np.asarray(sims))[-5:][::-1]
    #     best_captions = [captions[i] for i in top_indices]
    #     sims2 = clip_rank(device0, clip_model2, clip_preprocess2, raw_image, best_captions)
    #     best_index = np.argmax(np.asarray(sims2))
    if mode == 'CLIP-ViT-L' or mode == 'CLIP-RN50x64':
        sims = clip_rank(device1, clip_model1, clip_preprocess1, raw_image, captions)
        best_index = np.argmax(np.asarray(sims))
    elif mode == 'ITC' or mode == 'ITM':
        sims = blip_rank(device0, blip_ranking_model, raw_image, captions, mode=mode.lower())
        best_index = np.argmax(np.asarray(sims))

    print('ranking: ', mode)
    print('number of distinct sampled captions considered: ', len(captions))

    top_n = args.print_top_n
    top_indices = np.argsort(np.asarray(sims))[-top_n:][::-1]
    for i, j in enumerate(top_indices):
        print(f'{i} [{sims[j]:.4f}]: {captions[j]}')


if __name__ == '__main__':
    main()