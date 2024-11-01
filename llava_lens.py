import os
import sys
import warnings
warnings.simplefilter("ignore")
import difflib
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from hook_llava import load_hooked_llava


#   TODO:  for first N batches in MSCOCO2017 val, get the generated target logits list for each image
#          and compute the image patch that contains the highest logit alignment.  Then rerun with cache
#          on same images and plot target logit alignment of target image tokens across all layers.
#
#          Based on where a peak emerges, also perform MLP-out alignment on preceding layers to see
#          where the spike initiates (maybe do the target direction removal to see if this negatively)
#
#          Based on where that initiating layer is, retrieve MLP outs on target image tokens at that layer.
#          Take the mean and pass through the closest layer SAE on Mistral7b.  Get the indices for top k
#          latents and compute their dashboards for inspection.
#
#          If the Function Vector idea is true, I'd suspect that the FV direction is invariant to the target
#          logit, so the different logit directions will get canceled out in the mean, leaving the FV intact.
#
#          Alternatively, I can just pass some individual MLP_out - target_log_dir vectors to the SAE, or take
#          the mean of these "residual" vectors.
#
#          I also think it'd be interesting to do the above on 16_resid_post, as this is the vector that needs
#          to initiate the translation process.


def get_target_logits(vlm, hooked_model, data_iter, num_batches, batch_size):
    obj_det_question = "Give me a few objects and attributes found in this image. Return them comma separated, with nothing else."
    obj_det_prompt = f"USER: <image>\n{obj_det_question} ASSISTANT:"
    input_ids = tokenizer_image_token(obj_det_prompt, hooked_model.tokenizer, return_tensors='pt').to(device_str)
    input_ids = input_ids[None, :]
    input_tokens = torch.broadcast_to(input_ids, (batch_size, input_ids.shape[1]))
    pos_ids = torch.arange(0, input_tokens.shape[1], dtype=torch.long, device=input_tokens.device)
    pos_ids = torch.broadcast_to(pos_ids, (batch_size, pos_ids.shape[0]))

    all_embeds = []
    all_imgs = []
    for i in range(num_batches):
        imgs, _ = next(data_iter)
        imgs = imgs.to(device_str).half()
        all_imgs.append(imgs)

        _, _, _, _, embeds, _= vlm.prepare_inputs_labels_for_multimodal(input_tokens, pos_ids, None, None, None, imgs)
        all_embeds.append(embeds)

    vlm.cpu()
    # imgs = imgs.to('cpu')
    torch.cuda.empty_cache()

    #  Subtract off text tokens which is len (input_ids - BOS and <image>)
    # first_text_pos = embeds.shape[1] - (input_ids.shape[1] - 2)
    # center_patch_pos = ((first_text_pos - 1) // 2) + 1

    hooked_model.cuda()

    for i in range(num_batches):
        imgs = all_imgs[i]
        embeds = all_embeds[i]

        batch_eos = torch.zeros(batch_size, dtype=torch.bool)
        answers = torch.tensor([[]])
        answers = torch.broadcast_to(answers, (batch_size, 0))
        for j in range(16):
            logits = hooked_model.run_with_hooks(embeds, prepend_bos=False, start_at_layer=0, stop_at_layer=None, return_type='logits')
            # logits = llava_model(inputs_embeds=embeds, attention_mask=attn_mask, position_ids=pos_ids).logits
            # if j == 0:
            #     print("Top Logits on first pass")
            #     # print(torch.topk(logits[0, -1], 16))
            #     print(hooked_model.to_string(torch.topk(logits[0, -1], 16)[1]))

            next_tokens = torch.argmax(logits[:, -1], dim=-1).cuda()
            batch_eos[next_tokens == model.tokenizer.eos_token_id] = True
            if torch.all(batch_eos).item():
                break

            answers = torch.cat((answers, next_tokens[:, None].cpu()), dim=1)
            next_tokens = torch.broadcast_to(next_tokens[:, None], (embeds.shape[0], embeds.shape[1]+1))
            next_embeds = model.input_to_embed(next_tokens)[0]
            embeds = torch.cat((embeds, next_embeds[:, -1][:, None]), dim=1)

        # answers = torch.stack(answers, dim=1)
        np.save(f'/media/andrelongon/DATA/visual_llm_out/target_logits/batch_{i}.npy', answers.numpy())

        # outputs = model.to_string(answers)
        # out_capt = [s.split(model.tokenizer.eos_token)[0] for s in outputs]
        # print("Final Description")
        # print(out_capt[:10])


def get_target_tallies(vlm, hooked_model, logit_dir, data_iter, num_batches, batch_size):
    #   Get image token target logit alignments
    input_ids = tokenizer_image_token('USER: <image>\nDescribe the image. ASSISTANT:', hooked_model.tokenizer, return_tensors='pt').to(device_str)
    input_ids = input_ids[None, :]

    pos_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    pos_ids = torch.broadcast_to(pos_ids, (batch_size, pos_ids.shape[0]))
    input_tokens = torch.broadcast_to(input_ids, (batch_size, input_ids.shape[1]))
    
    all_embeds = []
    all_imgs = []
    for i in range(num_batches):
        imgs, _ = next(data_iter)
        imgs = imgs.to(device_str).half()
        all_imgs.append(imgs)

        _, _, _, _, embeds, _= vlm.prepare_inputs_labels_for_multimodal(input_tokens, pos_ids, None, None, None, imgs)
        all_embeds.append(embeds)

    llava_model.cpu()
    imgs = imgs.to('cpu')
    torch.cuda.empty_cache()

    #  Subtract off text tokens which is len (input_ids - BOS and <image>)
    first_text_pos = embeds.shape[1] - (input_ids.shape[1] - 2)
    # center_patch_pos = ((first_text_pos - 1) // 2) + 1

    hooked_model.cuda()

    all_tallies = []
    for i in range(num_batches):
        for j in range(batch_size):
            target_logits = np.load(f'{logit_dir}/batch_{i}.npy')
            target_logits = torch.tensor(target_logits[j], dtype=torch.long)
            targets = hooked_model.to_string(target_logits).split(hooked_model.tokenizer.eos_token)[0].strip()
            targets = targets.split(', ')

            if len(targets) < 5:
                continue
            else:
                targets = targets[:5]

            embeds = all_embeds[i][j][None, :, :]
            #   TODO:  can I run forward pass on GPU and cache everything to RAM?
            _, cache = hooked_model.run_with_cache(
                embeds,
                names_filter=lambda s: 'hook_resid_pre' in s,
                pos_slice=slice(1, first_text_pos, 1),
                device='cpu',
                prepend_bos=False,
                start_at_layer=0,
                stop_at_layer=None,
                return_type='logits'
            )

            layer_tallies = []
            #   For each output, count how many occurrences they show up for each patch, summed over layers.
            for k in cache:
                logits = hooked_model.unembed(torch.nn.functional.normalize(cache[k], dim=-1).cuda().float())

                patch_tallies = []
                for pos in range(logits.shape[1]):
                    top_logits = torch.topk(logits[0, pos], k=5, dim=-1)[1].cuda()
                    top_tokens = [hooked_model.to_string(l) for l in top_logits]

                    tallies = []
                    for t in targets:
                        count = 0
                        has_match = difflib.get_close_matches(t, top_tokens, n=1, cutoff=0.8)
                        if has_match:
                            count += 1
                        
                        tallies.append(count)
                    patch_tallies.append(tallies)
                layer_tallies.append(patch_tallies)
            all_tallies.append(layer_tallies)

    print(np.array(all_tallies).shape)
    np.save('/media/andrelongon/DATA/visual_llm_out/target_tallies/all_tallies.npy', np.array(all_tallies))


def inspect_alignments(vlm, hooked_model, logit_dir, talley_dir, data_iter, num_batches, batch_size, ablate_layer=None):
    all_tallies = np.load(talley_dir)
    # print(all_tallies.shape)
    # print(np.sum(all_tallies, axis=(0, 2, 3)))

    #   TODO:  sum across the logits and layers to get each patch's preferred target logit.
    #          Then obtain the logit alignment per layer.

    #   Get image token target logit alignments
    input_ids = tokenizer_image_token('USER: <image>\nDescribe the image. ASSISTANT:', hooked_model.tokenizer, return_tensors='pt').to(device_str)
    input_ids = input_ids[None, :]

    pos_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    pos_ids = torch.broadcast_to(pos_ids, (batch_size, pos_ids.shape[0]))
    input_tokens = torch.broadcast_to(input_ids, (batch_size, input_ids.shape[1]))
    
    all_embeds = []
    all_imgs = []
    for i in range(num_batches):
        imgs, _ = next(data_iter)
        imgs = imgs.to(device_str).half()
        all_imgs.append(imgs)

        _, _, _, _, embeds, _= vlm.prepare_inputs_labels_for_multimodal(input_tokens, pos_ids, None, None, None, imgs)
        all_embeds.append(embeds)

    llava_model.cpu()
    imgs = imgs.to('cpu')
    torch.cuda.empty_cache()

    #  Subtract off text tokens which is len (input_ids - BOS and <image>)
    first_text_pos = embeds.shape[1] - (input_ids.shape[1] - 2)
    # center_patch_pos = ((first_text_pos - 1) // 2) + 1

    all_aligns = []
    all_mean_resids = []
    hooked_model.cuda()
    for i in range(num_batches):
        for j in range(batch_size):
            if batch_size*i + j == all_tallies.shape[0]:
                break

            img_tallies = all_tallies[batch_size*i + j]
            patch_indices = np.where(np.sum(img_tallies, axis=(0, 2)) > 8)[0]
            if patch_indices.shape[0] == 0:
                continue

            targets = np.load(f'{logit_dir}/batch_{i}.npy')
            targets = torch.tensor(targets[j], dtype=torch.long)
            targets = hooked_model.to_string(targets).split(hooked_model.tokenizer.eos_token)[0].strip()
            targets = targets.split(', ')

            if len(targets) < 5:
                continue
            else:
                targets = targets[:5]

            target_tokens = hooked_model.to_tokens(targets)
            patch_targets = target_tokens[np.argmax(np.sum(img_tallies[:, patch_indices], axis=0), axis=1)]
            embeds = all_embeds[i][j][None, :, :]
            ablate_hook = []
            if ablate_layer is not None:
                #   Resids here refers to (mlp_out - target_direction) which will be used in downstream
                #   SAE latent inspection.
                def ablate_patches(value, hook):
                    # print(hook)
                    target_dirs = hooked_model.unembed.state_dict()['W_U'][:, patch_targets[:, 0]]
                    target_dirs = torch.transpose(target_dirs, 0, 1)
                    # target_norms = torch.linalg.vector_norm(target_dirs, dim=-1)
                    # target_dirs = torch.mul(target_dirs, 1 / target_norms[:, None].expand(-1, target_dirs.shape[1]))

                    aligns = torch.sum(torch.mul(value[0, patch_indices+1, :], target_dirs), dim=-1)
                    target_dirs = torch.mul(target_dirs, aligns[:, None].expand(-1, target_dirs.shape[1]))

                    # token_norms = torch.linalg.vector_norm(value[0, patch_indices+1, :], dim=-1)
                    # norm_ratios = token_norms / (target_norms + 1e-10)
                    # target_dirs = torch.mul(target_dirs, norm_ratios[:, None].expand(-1, target_dirs.shape[1]))

                    #   Add +1 to indices due to BOS being in pos=0.
                    # value[:, patch_indices+1, :] -= target_dirs
                    # all_mean_resids.append(np.mean(value[0, patch_indices+1, :].cpu().numpy(), axis=0))
                    all_mean_resids.append(np.mean(value[0, first_text_pos:, :].cpu().numpy(), axis=0))

                    # value[:, patch_indices+1, :] = 0.
                    return value

                ablate_hook = [(f'blocks.{ablate_layer}.hook_resid_pre', ablate_patches)]

            _, fwd, _ = hooked_model.get_caching_hooks()

            with hooked_model.hooks(fwd_hooks=fwd+ablate_hook):
                _, cache = hooked_model.run_with_cache(
                    embeds,
                    names_filter=lambda s: 'hook_mlp_out' in s,
                    pos_slice=slice(1, first_text_pos, 1),
                    device='cpu',
                    prepend_bos=False,
                    start_at_layer=0,
                    stop_at_layer=None,
                    return_type='logits'
                )
            
            mean_aligns = []
            for k in cache:
                logits = hooked_model.unembed(torch.nn.functional.normalize(cache[k], dim=-1).cuda().float())[0]
                aligns = torch.mean(logits[patch_indices, patch_targets[:, 0]])
                mean_aligns.append(aligns)

            all_aligns.append(mean_aligns)
        else:
            continue
        break

    np.save(f'/media/andrelongon/DATA/visual_llm_out/resids/layer{ablate_layer}_mean_txt_pre_out.npy', np.array(all_mean_resids))

    # mean_layer_aligns = torch.mean(torch.tensor(all_aligns), dim=0).numpy()

    # plt.figure(figsize=(12, 5))
    # plt.bar([f'{l}' for l in range(mean_layer_aligns.shape[0])], mean_layer_aligns, width=0.5)
    # plt.xlabel('Layers')
    # plt.ylabel('Alignment')
    # plt.title('Mean Logit Alignment for mlp_out = target_dir')
    # plt.savefig(f'/media/andrelongon/DATA/visual_llm_out/alignment_plots/mlp_out_eq_target_dir_{ablate_layer}_ablate.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablate_layer', type=int, required=False)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    device_str = f"cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device_str
    )

    num_batches = 4
    batch_size = 32
    IMAGE_SIZE = 336
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384),
        torchvision.transforms.CenterCrop(IMAGE_SIZE),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=clip_mean, std=clip_std)  #  Not needed as LLAVA performs normalization
    ])
    mscoco_ds = torchvision.datasets.ImageFolder('/media/andrelongon/DATA/mscoco2017_val/val2017', transform=transform)
    dataloader = torch.utils.data.DataLoader(mscoco_ds, batch_size=batch_size)
    data_iter = iter(dataloader)

    model = load_hooked_llava('/media/andrelongon/DATA/visual_llm_out/llava_mistral_lens.pth', tokenizer=tokenizer, device='cpu')
    print("HOOKED LLAVA LOADED")

    # get_target_logits(llava_model, model, data_iter, num_batches, batch_size)

    # get_target_tallies(llava_model, model, '/media/andrelongon/DATA/visual_llm_out/target_logits', data_iter, num_batches, batch_size)

    inspect_alignments(
        llava_model,
        model,
        '/media/andrelongon/DATA/visual_llm_out/target_logits',
        '/media/andrelongon/DATA/visual_llm_out/target_tallies/all_tallies.npy',
        data_iter,
        num_batches,
        batch_size,
        ablate_layer=args.ablate_layer
    )