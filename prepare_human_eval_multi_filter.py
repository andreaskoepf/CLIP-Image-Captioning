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
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=42, type=int, help='initialization of pseudo-RNG')
    parser.add_argument('--valid_json_path', default='/data/datasets/coco/annotations/captions_val2017.json', type=str)
    parser.add_argument('--image_folder_path', default='/data/datasets/coco/val2017/', type=str)
    parser.add_argument('--n', default=200, type=int)
    parser.add_argument('--id_prefix', default='Z', type=str)

    parser.add_argument('--output_folder', default='eval_Z', type=str)
    parser.add_argument('--params_json_fn', default='params.json', type=str)
    parser.add_argument('--top_k', default=2500, type=int)
    parser.add_argument('--force_eos_prob', default=0.9, type=float)
    parser.add_argument('--num_sampling_runs', default=1, type=int)

    parser.add_argument('--deviceA_index', default=0, type=int)
    parser.add_argument('--deviceB_index', default=1, type=int)

    parser.add_argument('--set_max_len', default=None, type=int)
    parser.add_argument('--set_min_len', default=None, type=int)
    parser.add_argument('--set_top_p', default=None, type=float)

    opt = parser.parse_args()
    return opt


def plot_histogram(data, label, title, x_label, y_label='Frequency', bins=100):
    plt.figure(figsize=(8,6))
    plt.hist(data, alpha=1.0, label=label, bins=bins)
    plt.xlabel(x_label, size=14)
    plt.ylabel(y_label, size=14)
    plt.title(title)
    plt.legend(loc='upper right')


def main():
    args = parse_args()

    seed = args.manual_seed
    torch.manual_seed(seed)

    random.seed(seed)

    torch.hub.set_dir('/mnt/sdb3/torch_hub')

    device0 = torch.device('cuda', args.deviceA_index)
    device1 = torch.device('cuda', args.deviceB_index)

    model,transform = load_blip_decoder(device1)

    clip_model_name1 = "ViT-L/14"
    clip_model_name2 = "RN50x64"
    print('loading CLIP1: ', clip_model_name1)
    clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device1)
    print('loading CLIP2: ', clip_model_name2)
    clip_model2, clip_preprocess2 = clip.load(clip_model_name2, device=device0)

    print('loading BLIP ranking model.')
    blip_ranking_model = load_blip_ranking_model(device0)
    
    # select partition of human annotations
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

    n = args.n
    randhalf = torch.randperm(n)[:n//2].sort().values
    gt = torch.zeros(n, dtype=torch.bool)
    gt[randhalf] = True

    gt_captions = [val_annotations[x] for x in torch.randperm(len(val_annotations))[:n]]
    for i,x in enumerate(tqdm(gt_captions)):

        caption = x.caption.lower()
        if caption[-1:] == '.':     # remove trailing full stop
            caption = caption[:-1]

        f = image_folder_path / x.image.file_name

        new_fn = output_image_folder_path / (uuid.uuid4().hex + '.jpg')
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

        CLIP_L_sim_threshold = 0.3
        CLIP_RN50x64_threshold = 0.3
        ITC_threshold = 0.55
        ITM_threshold = 0.99

        def filter_captions(captions, sims, threshold, default_p=0.1):
            results = [c for i,c in enumerate(captions) if sims[i] > threshold]
            if len(results) == 0:
                sorted_indices = np.argsort(np.asarray(sims))[::-1]
                num_select = max(1, int(len(captions) * default_p))
                sorted_indices = sorted_indices[:num_select]
            else:
                sims = [s for s in sims if s > threshold]
                sorted_indices = np.argsort(np.asarray(sims))[::-1]
            results = [captions[i] for i in sorted_indices]
            scores = [sims[i] for i in sorted_indices]
            return results, scores

        print('Candidates: ', len(captions))
        sims = clip_rank(device1, clip_model1, clip_preprocess1, raw_image, captions)
        captions0, scores0 = filter_captions(captions, sims, CLIP_L_sim_threshold)
        print(f'after CLIP L filtering: {len(captions0)}')

        sims = clip_rank(device0, clip_model2, clip_preprocess2, raw_image, captions0)
        captions1, scores1 = filter_captions(captions0, sims, CLIP_RN50x64_threshold)
        print(f'after RN filtering: {len(captions1)}')

        sims = blip_rank(device0, blip_ranking_model, raw_image, captions1, mode='itm')
        captions2, scores2 = filter_captions(captions1, sims, ITM_threshold)
        print(f'after ITM filtering: {len(captions2)}')

        sims = blip_rank(device0, blip_ranking_model, raw_image, captions2, mode='itc')
        captions3, scores3 = filter_captions(captions2, sims, ITC_threshold)
        print(f'after ITC filtering: {len(captions3)}')

        priv.append(
            {
                'id': args.id_prefix + f'{i:04d}',
                'file_name': str(rel_out_fn),
                'original_file_name': str(f),
                'ground_truth': caption,
                'synth_caption': captions3[0],
                'synth_candidates': len(captions),
                'image_size': [w, h]
            }
        )

        print('synth: ', captions3)
        print('human: ', caption)

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
            caption = entry['ground_truth' if gt[i] else 'synth_caption']
            print(f'<li><p><img src="{fn}" alt="{caption}" /><br />{id}: {caption}</p></li>', file=f)
        print('</ul></body></html>', file=f)


    # generate csv file for rating
    with open(output_folder_path / 'eval.csv', 'w', newline='') as f:
        w = csv.writer(f, dialect='excel')
        w.writerow(['id', 'file_name', 'caption', 'human', 'rating'])
        for i,entry in enumerate(priv):
            caption = entry['ground_truth' if gt[i] else 'synth_caption']
            w.writerow([entry['id'], entry['file_name'], caption, False, -1])


    # generate csv file with ground truth
    with open(output_folder_path / 'gt.csv', 'w', newline='') as f:
        w = csv.writer(f, dialect='excel')
        w.writerow(['id', 'file_name', 'human', 'human_caption', 'synth_caption', 'synth_candidates', 'original_file_name'])
        for i,entry in enumerate(priv):
            w.writerow([entry['id'], entry['file_name'], gt[i].item(), entry['ground_truth'], entry['synth_caption'], entry['synth_candidates'], entry['original_file_name']])


if __name__ == '__main__':
    main()
