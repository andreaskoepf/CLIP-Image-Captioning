import json
import csv
import math
import uuid
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from create_dataset import CocoJsonDataset
from sampling import clip_rank, load_blip_decoder, sample, blip_rank, load_blip_ranking_model
import clip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--valid_json_path', default='/data/datasets/coco/annotations/captions_val2017.json', type=str)
    parser.add_argument('--image_folder_path', default='/data/datasets/coco/val2017/', type=str)
    parser.add_argument('--n', default=200, type=int)
    parser.add_argument('--id_prefix', default='A', type=str)

    parser.add_argument('--output_folder', default='eval1', type=str)
    parser.add_argument('--params_json_fn', default='params.json', type=str)
    parser.add_argument('--top_k', default=2500, type=int)
    parser.add_argument('--force_eos_prob', default=0.9, type=float)
    parser.add_argument('--num_sampling_runs', default=1, type=int)

    parser.add_argument('--mode', default='CLIP-ViT-L+RN50x64', type=str)   # CLIP-ViT-L+RN50x64, CLIP-ViT-L, CLIP-RN50x64

    parser.add_argument('--deviceA_index', default=0, type=int)
    parser.add_argument('--deviceB_index', default=1, type=int)

    parser.add_argument('--set_max_len', default=None, type=int)
    parser.add_argument('--set_min_len', default=None, type=int)
    parser.add_argument('--set_top_p', default=None, type=float)

    opt = parser.parse_args()
    return opt


def main():
    args = parse_args()

    seed = args.manual_seed
    torch.manual_seed(seed)

    random.seed(seed)

    torch.hub.set_dir('/mnt/sdb3/torch_hub')

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

    # select partition of human annotations
    n = args.n
    randhalf = torch.randperm(n)[:n//2].sort().values
    gt = torch.zeros(n, dtype=torch.bool)
    gt[randhalf] = True


    val_annotations = CocoJsonDataset(args.valid_json_path)
    image_ids = list(val_annotations.image_by_id.keys())
    image_ids.sort()

    priv = []

    image_folder_path = Path(args.image_folder_path)

    top_k = args.top_k

    output_image_folder_name = 'images'

    output_folder_path = Path(args.output_folder)
    output_image_folder_path = output_folder_path / output_image_folder_name

    # create output image folder
    print('creating output directory: ', output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=False)

    print('creating image directory: ', output_image_folder_path)
    output_image_folder_path.mkdir(exist_ok=False)


    gt_captions = [val_annotations[x] for x in torch.randperm(len(val_annotations))[:n]]
    for i,x in enumerate(tqdm(gt_captions)):

        caption = x.caption.lower()
        if caption[-1:] == '.':     # remove trailing full stop
            caption = caption[:-1]

        f = image_folder_path / x.image.file_name

        new_fn = output_image_folder_path/ (uuid.uuid4().hex + '.jpg')
        shutil.copyfile(f, new_fn)

        rel_out_fn = new_fn.relative_to(output_folder_path)

        raw_image = Image.open(f).convert('RGB')   
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

        captions,_,_ = sample(
            image,
            model,
            sample_count=min_len.size(0),
            top_p=top_p,
            top_k=top_k,
            min_len=min_len,
            max_len=max_len,
            force_eos_log_prob=math.log(args.force_eos_prob),
            prompt='a picture of ',
            num_runs=args.num_sampling_runs)

        if mode == 'CLIP-ViT-L+RN50x64':
            sims = clip_rank(device1, clip_model1, clip_preprocess1, raw_image, captions)
            top_indices = np.argsort(np.asarray(sims))[-5:][::-1]
            best_captions = [captions[i] for i in top_indices]
            sims2 = clip_rank(device0, clip_model2, clip_preprocess2, raw_image, best_captions)
            best_index = np.argmax(np.asarray(sims2))
            synth_caption = best_captions[best_index]
        elif mode == 'CLIP-ViT-L' or mode == 'CLIP-RN50x64':
            sims = clip_rank(device1, clip_model1, clip_preprocess1, raw_image, captions)
            best_index = np.argmax(np.asarray(sims))
            synth_caption = captions[best_index]
        elif mode == 'ITC' or mode == 'ITM':
            sims = blip_rank(device0, blip_ranking_model, raw_image, captions, mode=mode.lower())
            best_index = np.argmax(np.asarray(sims))
            synth_caption = captions[best_index]

        #print('synth: ', synth_caption)
        #print('human: ', caption)

        priv.append(
            {
                'id': args.id_prefix + f'{i:04d}',
                'file_name': str(rel_out_fn),
                'original_file_name': str(f),
                'human_caption': caption,
                'synth_caption': synth_caption,
                'synth_candidates': len(captions),
                'image_size': [w, h]
            }
        )

    json_data = {
        'args': vars(args),
        'captions': priv
    } 


    with open(output_folder_path / args.params_json_fn, "w") as f:
        json.dump(json_data, f, indent=2)


    # generate html evaluation file
    with open(output_folder_path / 'eval.html', 'w') as f:
        print('<!DOCTYPE html>', file=f)
        print(f'<html><head><title>{str(output_folder_path)}</title>', file=f)
        print('<style>img { max-width: 512px; max-height: 512px; width: auto; height: auto; } li { margin-bottom: 75px; }</style></head><body>', file=f)
        print(f'<h1>{str(output_folder_path)}</h1>', file=f)
        print('<ul>', file=f)
        for i,entry in enumerate(priv):
            id = entry['id']
            fn = entry['file_name']
            caption = entry['human_caption' if gt[i] else 'synth_caption']
            print(f'<li><p><img src="{fn}" alt="{caption}" /><br />{id}: {caption}</p></li>', file=f)
        print('</ul></body></html>', file=f)


    # generate csv file for rating
    with open(output_folder_path / 'eval.csv', 'w', newline='') as f:
        w = csv.writer(f, dialect='excel')
        w.writerow(['id', 'file_name', 'caption', 'human', 'rating'])
        for i,entry in enumerate(priv):
            caption = entry['human_caption' if gt[i] else 'synth_caption']
            w.writerow([entry['id'], entry['file_name'], caption, False, -1])


    # generate csv file with ground truth
    with open(output_folder_path / 'gt.csv', 'w', newline='') as f:
        w = csv.writer(f, dialect='excel')
        w.writerow(['id', 'file_name', 'human', 'human_caption', 'synth_caption', 'synth_candidates', 'original_file_name'])
        for i,entry in enumerate(priv):
            w.writerow([entry['id'], entry['file_name'], gt[i].item(), entry['human_caption'], entry['synth_caption'], entry['synth_candidates'], entry['original_file_name']])


if __name__ == '__main__':
    main()
