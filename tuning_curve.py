import os
import sys
from pathlib import Path
import copy
import argparse
import torch
import torchvision
from torchvision import models, transforms, utils
import numpy as np
from safetensors.torch import load_file
from sae_lens import SAE

sys.path.append('LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from imnet_val import get_imnet_val_acts
from model_wrapper import ModelWrapper


def saveTopN(imgs, lista, neuron_id, n=9, path=""):
    topil = transforms.ToPILImage()
    
    neuron_path = os.path.join(path, neuron_id)
    Path(neuron_path).mkdir(exist_ok=True)

    grids_path = os.path.join(path, "all_grids")
    Path(grids_path).mkdir(exist_ok=True)

    exc_imgs = []
    for i in range(n):
        img = imgs[int(lista[i])]
        exc_imgs.append(img)
        img = topil(img)
        img.save(os.path.join(neuron_path, f"{i}.png"))

    grid = utils.make_grid(exc_imgs, nrow=3)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(os.path.join(grids_path, f"{neuron_id}.png"))


def create_tuning_curves(model, module_name, savedir, valdir, device='cpu'):

    all_images, act_list, unrolled_acts, all_act_list, all_ord_sorted, _ = \
        get_imnet_val_acts(model, valdir, sort_acts=False, device=device)

    unrolled_acts = np.array(unrolled_acts)
    if len(unrolled_acts.shape) == 2:
        unrolled_acts = np.transpose(unrolled_acts, (1, 0))

        for i in range(unrolled_acts.shape[0]):
            unrolled_act = unrolled_acts[i].tolist()
            all_ord_list = np.arange(len(all_images)).tolist()
            all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))

            saveTopN(all_images, all_ord_sorted, f"{module_name}_neuron{i}", path=savedir)

            np.save(os.path.join(savedir, f"{module_name}_unit{i}_unrolled_act.npy"),
                    np.array(unrolled_act))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_act_list.npy"),
                    np.array(list(all_act_list)))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_ord_sorted.npy"),
                    np.array(list(all_ord_sorted)))

            if i == 256:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--module', type=str)
    parser.add_argument('--sae', action='store_true')
    parser.add_argument('--use_saelens', action='store_true')
    parser.add_argument('--sae_root', type=str, required=False)
    parser.add_argument('--imnet_val_dir', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    savedir = os.path.join(args.savedir, args.network, args.module)
    Path(savedir).mkdir(parents=True, exist_ok=True)

    device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device_str
    )

    input_ids = tokenizer_image_token('What is this an image of?<image>', tokenizer, return_tensors='pt').to(device_str)

    if args.sae:
        if args.use_saelens:
            sae, _, _  = SAE.from_pretrained(
                release = "mistral-7b-res-wg",
                sae_id = f"{args.module}.hook_resid_pre",
                device = device_str
            )
            sae_weights = sae.state_dict()
            expansion = 16
        else:
            sae_weights = load_file(os.path.join(args.sae_root, "sae_weights.safetensors"))
            expansion = 8

        model = ModelWrapper(model, input_ids, image_processor, int(args.module.split('.')[-1]), args.sae, expansion=expansion)
        model.to(device_str).eval()

        states = model.state_dict()
        states['map.weight'] = sae_weights['W_dec']
        model.load_state_dict(states)

    create_tuning_curves(model, args.module, savedir, args.imnet_val_dir, device=device_str)