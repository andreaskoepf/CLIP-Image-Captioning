import readline
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from BLIP.models.blip import blip_decoder
from models.blip_itm import blip_itm


def load_demo_image(image_size,device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image, raw_image



def load_blip_ranking_model(device, image_size=384):
    blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
    blip_model = blip_itm(pretrained=blip_model_url, image_size=image_size, vit='large', med_config='BLIP/configs/med_config.json')
    blip_model.eval()
    blip_model.to(device)
    return blip_model


@torch.no_grad()
def blip_rank(device, model_blip, image_pil, text_list, image_size=384, mode="itc"):
    similarities= []

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(image_pil).unsqueeze(0).to(device)   

    for caption in text_list:
        if mode == 'itm':
            itm_output = model_blip(image, caption, match_head='itm')
            itm_score = F.softmax(itm_output, dim=1)[:,1]
            similarities.append(itm_score.item())
        elif mode == 'itc':
            itc_score = model_blip(image, caption, match_head='itc')
            similarities.append(itc_score.item())
        else:
            raise RuntimeError(f'blip ranking mode "{mode}" not supported')

    return similarities


def main():
    torch.hub.set_dir('/mnt/sdb3/torch_hub')

    device = torch.device('cuda', 1)

    print('loading ranking model')
    blip_ranking_model = load_blip_ranking_model(device)

    image_size = 384
    image, raw_image = load_demo_image(image_size=image_size, device=device)
    print('loaded image:', image.shape)


    with torch.no_grad():
        image_embeds = blip_ranking_model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  

    tokenizer = blip_ranking_model.tokenizer
    # text = blip_ranking_model.tokenizer('This is a simple test', padding='max_length', truncation=True, max_length=35, 
    #                           return_tensors="pt").to(image.device)
    # print('text', text.attention_mask, text.attention_mask.shape)
    # quit()

    # get vocabulary size
    bert_embedding = blip_ranking_model.text_encoder.embeddings
    word_embeddings = bert_embedding.word_embeddings
    print('pos embedding type:', bert_embedding.position_embedding_type)
    print('word_embeddings', word_embeddings.weight.shape)


    seq_length = 35
    fake_logits = torch.randn(1, seq_length, word_embeddings.weight.size(0), device=device) * 0.01
    fake_logits.requires_grad = True
    attention_mask = torch.ones(1, seq_length, device=device)

    optimizer = torch.optim.AdamW(params=[fake_logits], lr=0.1, weight_decay=0)

    tau_start = 5.0
    tau_end = 0.001
    t_max = 5000
    for i in range(t_max):

        t = i / (t_max-1)

        tau = (1-t) * tau_start + t * tau_end

        p = F.gumbel_softmax(fake_logits, tau=tau, hard=False)
         
        txt_embed = p @ word_embeddings.weight

        position_ids = bert_embedding.position_ids[:, :seq_length]

        position_embeddings = bert_embedding.position_embeddings(position_ids)
        txt_embed += position_embeddings
        txt_embed = bert_embedding.LayerNorm(txt_embed)
        

        text_output = blip_ranking_model.text_encoder.forward(
            encoder_embeds=txt_embed, 
            #attention_mask=attention_mask,
            return_dict = True,
            mode = 'text'
        )

        image_feat = F.normalize(blip_ranking_model.vision_proj(image_embeds[:,0,:]), dim=-1)   
        text_feat = F.normalize(blip_ranking_model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
        sim = image_feat @ text_feat.t()
        loss = -sim

        # optionally include ITM
        # text_output2 = blip_ranking_model.text_encoder.forward(
        #     encoder_embeds=txt_embed, 
        #     #attention_mask=attention_mask,
        #     encoder_hidden_states = image_embeds,
        #     encoder_attention_mask = image_atts,
        #     return_dict = True
        # )

        # itm_output = blip_ranking_model.itm_head(text_output2.last_hidden_state[:,0,:])
        # #print('itm_output', itm_output[0, 1] - itm_output[0, 0])

        # # take ITM loss + ITC
        # loss += itm_output[0, 0] - itm_output[0, 1]
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+1) % 100 == 0:
            r = p.argmax(dim=-1)[0].tolist()
            caption = tokenizer.decode(r, skip_special_tokens=False)
            print(f'[{i}] tau: {tau:.2f}; sim: {sim.item():.4f}; itm: {itm_output[0, 1]:.4f}; {caption}')
            sim_rank = blip_rank(device, blip_ranking_model, raw_image, [caption])
            print('sim_rank', sim_rank)
            

    

    #torch.rand()

    #text_output = blip_ranking_model.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict = True, mode = 'text') 



    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    
    # model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base', med_config='BLIP/configs/med_config.json')
    # model.eval()
    # model = model.to(device)

    # feat = model.visual_encoder(image)


if __name__ == '__main__':
    main()