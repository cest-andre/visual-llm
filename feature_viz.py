from pathlib import Path
import os
import sys
import argparse
import warnings
warnings.simplefilter("ignore")
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param
from safetensors.torch import load_file
from sae_lens import SAE

sys.path.append('LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token

from model_wrapper import ModelWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--basedir', type=str)
parser.add_argument('--module', type=str)
parser.add_argument('--start_feat', type=int, default=0)
parser.add_argument('--stop_feat', type=int, default=8)
parser.add_argument('--sae', action='store_true')
parser.add_argument('--use_saelens', action='store_true')
parser.add_argument('--sae_root', type=str, required=False)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=24)
parser.add_argument('--steps', type=int, default=2560)
args = parser.parse_args()

device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

model_path = "liuhaotian/llava-v1.6-mistral-7b"
tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device_str,
    load_8bit=True
)

input_ids = tokenizer_image_token('<image>A picture of', tokenizer, return_tensors='pt').to(device_str)

units = None
model = None
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
        sae_sparsity = load_file(os.path.join(args.sae_root, "sparsity.safetensors"))['sparsity']
        expansion = 8
        units = torch.nonzero(sae_sparsity > -5)[args.start_feat:args.stop_feat][:, 0].tolist()
        sae_weights = load_file(os.path.join(args.sae_root, "sae_weights.safetensors"))

    model = ModelWrapper(llava_model, input_ids, image_processor, int(args.module.split('.')[-1]), args.sae, expansion=expansion)
    model.to(device_str).eval()
    states = model.state_dict()
    states['map.weight'] = sae_weights['W_dec']
    model.load_state_dict(states)
else:
    model = ModelWrapper(llava_model, input_ids, image_processor, int(args.module.split('.')[-1]), args.sae)
    model.to(device_str).eval()

if units is None:
    units = list(range(args.start_feat, args.stop_feat))

# clip_preprocess = lambda x: image_processor.preprocess(x, do_rescale=False, return_tensors='pt')['pixel_values']

clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
clip_norm = transforms.Normalize(mean=clip_mean, std=clip_std)
transforms = None
if args.jitter < 4:
    transforms = [
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        torchvision.transforms.CenterCrop(336)
    ]
else:
    transforms = [
        transform.pad(args.jitter),
        transform.jitter(args.jitter),
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        transform.jitter(int(args.jitter/2)),
        torchvision.transforms.CenterCrop(336),
        clip_norm
    ]

for unit in units:
    savedir = os.path.join(args.basedir, args.network, args.module, f"unit{unit}")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    model.target_neuron = unit

    print(f"BEGIN MODULE {args.module} NEURON {unit}")
    param_f = lambda: param.images.image(336, decorrelate=True)
    obj = objectives.neuron('0', unit)

    imgs = render.render_vis(nn.Sequential(model), obj, param_f=param_f, transforms=transforms, thresholds=(args.steps,), preprocess=False, show_image=False)
    img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
    img.save(os.path.join(savedir, f"{args.steps}steps_distill_mean.png"))

    np.save(os.path.join(savedir, f"{args.steps}steps_distill_mean.npy"), np.array(model.all_acts))

    model.all_acts = []