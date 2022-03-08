from pathlib import Path
import hashlib
import pickle
import re
from tqdm import tqdm
import json

from create_dataset import CocoJsonDataset


def create_index(path):
    file_to_hash = {}
    hash_to_files = {}

    image_files = [
        *path.glob("**/*.png"),
        *path.glob("**/*.jpg"),
        *path.glob("**/*.jpeg"),
        *path.glob("**/*.bmp"),
    ]
    print('found: ', len(image_files) )

    for i, fn in enumerate(tqdm(image_files)):
        data = fn.read_bytes()
        h = hashlib.new('sha256')
        h.update(data)
        
        hash = h.hexdigest()

        file_to_hash[fn] = hash
        if hash in hash_to_files:
            hash_to_files[hash].append(fn)
        else:
            hash_to_files[hash] = [fn]

    return file_to_hash, hash_to_files  


def store_index(path, out_filename):
    file_to_hash, hash_to_files = create_index(Path(path))
    index = { 'file_to_hash': file_to_hash, 'hash_to_file': hash_to_files }
    with open(out_filename, 'wb') as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_index(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def compute_file_hashes():
    store_index('/data/datasets/coco/train2017/', 'coco_train2017.index.pickle')
    store_index('/data/datasets/coco/val2017/', 'coco_val2017.index.pickle')
    store_index('/mnt/sdb3/clip_finetune/', 'clip_finetune.index.pickle')


def add_dir_to_index(in_filename, dir_path, out_filename):
    index = load_index(in_filename)
    print('unique files before:', len(index['hash_to_file']))

    file_to_hash, hash_to_files  = create_index(Path(dir_path))

    idx_file_to_hash = index['file_to_hash']
    for fn,hash in file_to_hash.items():
        idx_file_to_hash[fn] = hash
    
    idx_hash_to_files = index['hash_to_file']
    for hash,fn_list in hash_to_files.items():
        if hash in idx_hash_to_files:
            idx_hash_to_files[hash] = idx_hash_to_files[hash] + fn_list
        else:
            idx_hash_to_files[hash] = fn_list

    print('unique files after:', len(index['hash_to_file']))
    with open(out_filename, 'wb') as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_caption(s):
    y = s.split('\n')
    s2 = ''.join(y)
    if len(s2) <= len(y):    # if string consisted of <1-character lines
        return s2
    return re.sub("\s+", " ", s).strip()


def merge_captions():
    annotation_json_path = '/data/datasets/coco/annotations/captions_train2017.json'
    train_annotations = CocoJsonDataset(annotation_json_path)

    train_index = load_index('coco_train2017.index.pickle')
    train_hash_index = train_index['hash_to_file']
    train_file_index = train_index['file_to_hash']

    coco_captions_by_hash = {}
    coco_image_dir = Path('/data/datasets/coco/train2017/')
    
    for entry in train_annotations:
        caption = normalize_caption(entry.caption)
        source_path = coco_image_dir / entry.image.file_name
        if source_path in train_file_index:
            hash = train_file_index[source_path]
            if hash in coco_captions_by_hash:
                coco_captions_by_hash[hash].append(caption)
            else:
                coco_captions_by_hash[hash] = [caption]

    #train_annotations
    val_hash_index = load_index('coco_val2017.index.pickle')['hash_to_file']
    finetune_hash_index = load_index('clip_finetune2.index.pickle')['hash_to_file']

    images = []
    annotations = []

    next_image_id = 0
    next_caption_id = 0
    base_path = '/mnt/clip_finetune'
    #redirect_path = Path('/mnt/sdb3/clip_finetune')
    redirect_path = None
    for hash, fns in tqdm(finetune_hash_index.items()):

        # 1. ignore if part of validation set
        if hash in val_hash_index:
            continue

        captions = []
        for fn in fns:
            # try to load .txt file caption
            txt_file_path = fn.parent / (fn.stem + '.txt')
            if redirect_path is not None:
                txt_file_path = redirect_path / txt_file_path.relative_to(base_path)
            if txt_file_path.is_file():
                txt = txt_file_path.read_text()
                c = normalize_caption(txt)
                if c not in captions:
                    captions.append(c)
                    #print(c)

        # 2. check if part of coco, add missing captions
        if hash in coco_captions_by_hash:
            coco_captions = coco_captions_by_hash[hash]
            for c in coco_captions:
                if c not in captions:
                    captions.append(c)
        
        fns.sort()
        # generate image entry
        image_id = next_image_id
        next_image_id += 1
        images.append({
            'file_name': str(fns[0].relative_to(base_path)),
            'id': image_id
        })

        # generate caption entries
        for c in captions:
            caption_id = next_caption_id
            next_caption_id += 1
            annotations.append({
                'image_id': image_id,
                'id': caption_id,
                'caption': c
            })

    json_data = {
        'images': images,
        'annotations': annotations
    }

    output_json_path = 'test5.json'
    print('writing:', output_json_path)
    with open(output_json_path, "w") as f:
        json.dump(json_data, f)
    print('done.')




def main():
    #add_dir_to_index('clip_finetune.index.pickle', '/mnt/clip_finetune/00080', 'clip_finetune2.index.pickle')
    
    merge_captions()
    quit()

    #store_index('/mnt/sdb3/clip_finetune', 'clip_finetune_check.index.pickle')
    #quit()

    #train_index = load_index('coco_train2017.index.pickle')['hash_to_file']
    #val_index = load_index('coco_val2017.index.pickle')['hash_to_file']
    finetune_index = load_index('clip_finetune.index.pickle')['hash_to_file']
    #finetune_files = load_index('clip_finetune.index.pickle')['file_to_hash']
    # print('#finetune_index:', len(finetune_index))
    # print('#train2017_index:', len(train_index))
    # print('#val2017_index:', len(val_index))
    
    # val_in_finetune_keys = val_index.keys() & finetune_index.keys()
    # print('coco val2017 in clip_finetune:', len(val_in_finetune_keys))
    # train_in_finetune_keys = train_index.keys() & finetune_index.keys()
    # print('coco train2017 in clip_finetune:', len(train_in_finetune_keys))

    #missing_in_finetune = set(train_index.keys()).difference(set(finetune_index.keys()))
    #print('missing train2017 entries in clip_finetune', len(missing_in_finetune))
    # dst_folder = '/mnt/sdb3/clip_finetune/00080/'
    # for k in missing_in_finetune:
    #     src_fn = train_index[k]
    #     print(f'cp {str(src_fn[0])} {dst_folder}')
        

    # remove all elements from coco val2017 set
    #val_in_finetune_keys = val_index.keys() & finetune_index.keys()
    # print('coco val2017 in clip_finetune:', len(val_in_finetune_keys))
    # relative_base = '/mnt/clip_finetune/'
    # for k in val_in_finetune_keys:
    #     for fn in finetune_index[k]:
    #         path = fn.relative_to(relative_base)
    #         parent = path.parent
    #         print(f'rm {parent.joinpath(path.stem + ".*")}')


    #print('clip_finetune', len(finetune_files))
    dupes = [(x, len(x)) for x in finetune_index.values() if len(x) > 1] 
    print('#dupe sha265 hashes in clip_finetune', len(dupes))
    max_dupe = max(dupes, key=lambda x: x[1])
    print(max_dupe)

    # collect different captions per image hash

    # remove all dupes and val overlap
    
    

if __name__ == '__main__':
    main()
