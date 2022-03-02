import time
from typing import Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import clip
from models.blip import blip_decoder
import glob


def cos_sim(a, b):
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return a @ b.T


@torch.no_grad()
def clip_rank(device, clip_model, preprocess, image_pil, text_list):

    similarities= []
    image = preprocess(image_pil).unsqueeze(0).to(device)

    image_features = clip_model.encode_image(image)

    for txt in text_list:
        text_tokens = clip.tokenize(txt, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        s = cos_sim(text_features, image_features).item()
        similarities.append(s)

    return similarities


def repetition_penalty_apply(logits, tokens, penalty):
    tok_logits = torch.gather(logits, -1, tokens)
    tok_logits = torch.where(tok_logits < 0, tok_logits * penalty, tok_logits / penalty)
    logits.scatter_(-1, tokens, tok_logits)
    return logits


def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    batch_size = logits.size(0)
    num_logits = logits.size(-1)
    device = logits.device
    #print('top_k', type(top_k), top_k)
    if type(top_k) == float:
        if top_k > 0 and top_k < 1:
            top_k = max(1, int(top_k * num_logits))
        else:
            top_k = int(top_k)
    # Remove all tokens with a probability less than the last token of the top-k
    if type(top_k) == int:
        if top_k > 0:
            cutoff = torch.topk(logits, k=top_k, largest=True).values[:, -1:]
            indices_to_remove = logits < cutoff
            logits[indices_to_remove] = filter_value      
    elif torch.any(top_k > 0):
        assert top_k.size(0) == batch_size
        top_k = top_k.clamp_max(num_logits)
        for i in range(batch_size):
            k = top_k[i] 
            if k <= 0:
                continue
            if k < 1:
                k = max(1, int(k * num_logits))
            cutoff = torch.topk(logits[i], k=k, largest=True).values[-1]
            indices_to_remove = logits[i] < cutoff
            logits[i][indices_to_remove] = filter_value
    if type(top_p) == float and top_p > 0.0 or torch.any(top_p > 0):
        if type(top_p) == torch.Tensor and top_p.size(-1) != 1:
            top_p = top_p.unsqueeze(-1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        # convert sorted indices into flat indices
        row_starts = torch.arange(sorted_indices.shape[0], device=device).unsqueeze(1) * sorted_indices.shape[1]
        sorted_indices_flat = sorted_indices + row_starts
        indices_to_remove = sorted_indices_flat[sorted_indices_to_remove]
        logits = logits.contiguous()
        logits.view(-1)[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def generate(
    model,
    inputs: Optional[torch.Tensor],
    encoder_hidden_states,
    encoder_attention_mask,
    eos_token_id,
    top_p,
    top_k,
    min_length,
    max_length,
    repetition_penalty: Optional[float] = None,
):
    # run until max or no candidates remaining
    total_max_length = max_length.max()

    results = []

    for i in range(total_max_length):
        if inputs.size(0) == 0:
            break

        outputs = model.forward(
            input_ids = inputs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
        )
        logits = outputs['logits']      # [1, 4, 30524]

        last_token_logits = logits[:,-1,:]
        min_indices = torch.nonzero(i < min_length).view(-1)
        last_token_logits[min_indices, eos_token_id] = float('-inf')

        if repetition_penalty is not None and repetition_penalty > 0:
            repetition_penalty_apply(last_token_logits, tokens=inputs, penalty=repetition_penalty)

        last_token_logits = top_k_top_p_filtering_batch(last_token_logits, top_p=top_p, top_k=top_k)
        
        p = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(p, 1)

        completed = torch.logical_or(next_token.squeeze(-1) == eos_token_id, max_length <= i)
        if torch.any(completed):
            results.append(inputs[completed])
            not_completed = torch.logical_not(completed)

            inputs = inputs[not_completed]
            next_token = next_token[not_completed]
            if type(top_p) == torch.Tensor:
                top_p = top_p[not_completed]
            if type(top_k) == torch.Tensor: 
                top_k = top_k[not_completed]
            min_length = min_length[not_completed]
            max_length = max_length[not_completed]
            encoder_hidden_states = encoder_hidden_states[not_completed]
            encoder_attention_mask = encoder_attention_mask[not_completed]

        inputs = torch.cat([inputs, next_token], dim=-1)

    if inputs.size(0) > 0:
        results.append(inputs)

    return results


@torch.no_grad()
def sample(image, blip_model, sample_count=3, top_p=0, top_k=0, min_len=0, max_len=0, prompt='a picture of '):
    batch_size = image.size(0)
    device = image.device
    image_embeds = blip_model.visual_encoder(image)

    image_embeds = image_embeds.repeat_interleave(sample_count, dim=0)    
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

    bos_token_id = blip_model.tokenizer.bos_token_id
    eos_token_id = blip_model.tokenizer.sep_token_id
    
    prompt_ = [prompt] * batch_size
    input_ids = blip_model.tokenizer(prompt_, return_tensors="pt").input_ids.to(device)
    input_ids[:,0] = bos_token_id  # replace begin token 
    input_ids = input_ids[:, :-1]   # remove end token
    input_ids = input_ids.repeat_interleave(sample_count, dim=0)

    outputs = generate(blip_model.text_decoder, input_ids, image_embeds, image_atts, 
        eos_token_id=eos_token_id,
        top_p=top_p,
        top_k=top_k,
        min_length=min_len,
        max_length=max_len,
        repetition_penalty=1.4)

    captions = []  
    for output in outputs:
        for o in output:
            caption = blip_model.tokenizer.decode(o, skip_special_tokens=True)
            captions.append(caption[len(prompt):])  # remove prompt
    return captions


def main():
    torch.hub.set_dir('/mnt/sdb3/torch_hub')

    device = torch.device('cuda', 1)
    device0 = torch.device('cuda', 0)

    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    
    model = blip_decoder(pretrained=model_url, image_size=384, vit='large', med_config='BLIP/configs/med_config.json')
    model.eval()
    model = model.to(device)

    clip_model_name1="ViT-L/14"
    clip_model1, clip_preprocess1 = clip.load(clip_model_name1, device=device)

    clip_model_name2="RN50x64"
    clip_model2, clip_preprocess2 = clip.load(clip_model_name2, device=device0)
   
    best_parameters = []

    files = glob.glob("./images/image-photo/*.jpg")
    for f in files[:40]:
        print(f)
        raw_image = Image.open(f).convert('RGB')   
        w,h = raw_image.size

        image = transform(raw_image).unsqueeze(0).to(device)     
    
        top_k = 5000
        top_p = 0.3
        #top_p = torch.tensor(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]*4), device=device)
        #top_p = torch.tensor(([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) * 2, device=device)
        min_len = torch.tensor(([10]*7 + [15]*7 + [25]*7 + [30]*7), device=device)
        max_len = torch.tensor(([30]*7 + [30]*7 + [45]*7 + [45]*7), device=device)

        # print('top_k', top_k)
        # print('top_p', top_p)
        # print('min_len', min_len)
        # print('max_len', max_len)

        start = time.time()
        captions = sample(image, model, sample_count=min_len.size(0), top_p=top_p, top_k=top_k, min_len=min_len, max_len=max_len, prompt='a picture of ')
        duration = time.time() - start
        
        for i,c in enumerate(captions):
            print(i, c)
        print(f'took: {duration:.2f}s')

        sims = clip_rank(device, clip_model1, clip_preprocess1, raw_image, captions)
        print('sims:', sims)
        top_indices = np.argsort(np.asarray(sims))[-5:]
        argmax_ = top_indices[-1]
        best_captions = [captions[i] for i in top_indices]

        print('Filtered:')
        for i in range(len(best_captions)):
            print(f'{i}: {best_captions[-i-1]}')

        sims2 = clip_rank(device0, clip_model2, clip_preprocess2, raw_image, best_captions)
        best_index = np.argmax(np.asarray(sims2))
        print('top1:', best_index)
        print(best_captions[best_index])

        duration2 = time.time() - start
        print(f'took (incl. clip filtering): {duration2:.2f}s')

        #model.generate(image, sample=True, num_beams=64, max_length=30, min_length=10, top_p=topP, repetition_penalty=rep_pen)
        continue

        captions = []

        rep_pen = 1.4

        with torch.no_grad():
            for topP in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                #[0.05,0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]

                caption = model.generate(image, sample=True, num_beams=3, max_length=30, min_length=10, top_p=topP, repetition_penalty=rep_pen)
                #def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0)
                if caption not in [x[0] for x in captions]:
                    print('caption:', caption)
                    captions.append((caption, {'sample': True, 'num_beams': 3, 'max_length': 30, 'min_length': 10, 'top_p': topP}))

            for beam_n in [1,2,3,4,5,6,7,8]:
                #[0.05,0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]

                caption = model.generate(image, sample=False, num_beams=beam_n, max_length=30, min_length=10, top_p=0.9, repetition_penalty=rep_pen)
                if caption not in [x[0] for x in captions]:
                    print('caption:', caption)
                    captions.append((caption, {'sample': False, 'num_beams': beam_n, 'max_length': 30, 'min_length': 10, 'top_p': 0.9}))

            for topP in [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7]:
                #[0.05,0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]

                caption = model.generate(image, sample=True, max_length=45, min_length=30,top_p=topP,repetition_penalty=rep_pen)
                if caption not in [x[0] for x in captions]:
                    print('caption:', caption)
                    #def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0)
                    captions.append((caption, {'sample': True, 'num_beams': 3, 'max_length': 45, 'min_length': 30, 'top_p': topP}))

            for beam_n in [1,2,3,4,5,6,7,8]:
                #[0.05,0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]

                caption = model.generate(image, sample=False, num_beams=beam_n, max_length=45, min_length=30,repetition_penalty=rep_pen)
                if caption not in [x[0] for x in captions]:
                    print('caption:', caption)
                    #def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0)
                    captions.append((caption, {'sample': False, 'num_beams': beam_n, 'max_length': 45, 'min_length': 30, 'top_p': 0.9}))

            best_candidates = []
            sims = clip_rank(device, clip_model, clip_preprocess, raw_image, [x[0] for x in captions])
            print('sims:', sims)
            top3_indices = np.argsort(np.asarray(sims))[-3:]
            argmax_ = top3_indices[-1]
            print('top3:', top3_indices)
            
            print("Caption with highest sim", captions[argmax_][0])
            best_candidates.append(captions[argmax_][0])
            best_parameters.append(captions[argmax_][1])

            # #print(sims[argmax_])
            # del sims[argmax_]
            # del captions[argmax_]
            # argmax_ = np.argmax(np.asarray(sims))
            # #print("Caption with 2nd highest sim")
            # #print (captions[argmax_][0])
            # best_cannidates.append(captions[argmax_][0])
            # #print(sims[argmax_])
            # del sims[argmax_]
            # del captions[argmax_]
            # argmax_ = np.argmax(np.asarray(sims))
            # #print("Caption with 3nd highest sim")
            # #print (captions[argmax_][0])
            # best_cannidates.append(captions[argmax_][0])
            # del sims[argmax_]
            # del captions[argmax_]
            # argmax_ = np.argmax(np.asarray(sims))
            # #print("Caption with 3nd highest sim")
            # #print (captions[argmax_][0])
            # best_cannidates.append(captions[argmax_][0])
            # #print(sims[argmax_])

            # sims= clip_rank(raw_image, best_cannidates, clip_model="RN50x64")
            
            # argmax_ = np.argmax(np.asarray(sims))
            # print("BEST CAPTION AFTER RANKING WITH CLIP ViT L 14 & RESNET50x64:")
            # print (best_cannidates[argmax_])

        print('best_parameters', best_parameters)

        
if __name__ == '__main__':
    main()