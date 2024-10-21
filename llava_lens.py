import os
import sys
import warnings
warnings.simplefilter("ignore")
import torch
import torchvision
import numpy as np

sys.path.append('LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from hook_llava import load_hooked_llava


@torch.autocast(device_type="cuda")
def get_llava_embeds(num_batches, device_str):
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device_str
    )
    input_ids = tokenizer_image_token('<image>A picture of', tokenizer, return_tensors='pt').to(device_str)
    input_ids = input_ids[None, :]

    batch_size = 1
    IMAGE_SIZE = 336
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384),
        torchvision.transforms.CenterCrop(IMAGE_SIZE),
        torchvision.transforms.ToTensor(),
    ])
    clip_norm = torchvision.transforms.Normalize(mean=clip_mean, std=clip_std)

    imagenet_data = torchvision.datasets.ImageFolder('~/imagenet/val', transform=transform)
    sampler = torch.utils.data.RandomSampler(imagenet_data, num_samples=batch_size)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, drop_last=False, sampler=sampler)

    for j, (inputs, labels) in enumerate(dataloader):
        if j == num_batches:
            break

        imgs = inputs.to(device_str)

        pos_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        pos_ids = torch.broadcast_to(pos_ids, (imgs.shape[0], pos_ids.shape[0]))
        input_tokens = torch.broadcast_to(input_ids, (imgs.shape[0], input_ids.shape[1]))
        _, _, _, _, x, _ = llava_model.prepare_inputs_labels_for_multimodal(input_tokens, pos_ids, None, None, None, imgs)

    first_text_pos = x.shape[1] - (input_ids.shape[1] - 2)
    # center_patch_pos = ((first_text_pos - 1) // 2) + 1

    return first_text_pos, x


# torch.set_grad_enabled(False)
# device_str = f"cuda:0" if torch.cuda.is_available() else "cpu"

# # first_text_pos, embeds = get_llava_embeds(1, device_str)
# # np.save('/home/ajl_onion123/results/token_embeds/test.npy', embeds.cpu().detach().numpy())

# # exit()

# embeds = np.load('/home/ajl_onion123/results/token_embeds/test.npy')
# embeds = torch.tensor(embeds, device=device_str)

# model = load_hooked_llava('/home/ajl_onion123/weights/llava_mistral_lens.pth', device='cpu')
# print("HOOKED LLAVA LOADED")
# model.cuda()
# #   TODO:  can I run forward pass on GPU and cache everything to RAM?
# logits, cache = model.run_with_cache(embeds, prepend_bos=False, start_at_layer=1, stop_at_layer=8, return_type='logits')

# print(cache['blocks.1.attn.hook_attn_scores'].shape)
# print(cache['blocks.1.attn.hook_attn_scores'][0, :4, first_text_pos-4:, first_text_pos-4:])