from pathlib import Path
import os
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import models
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param
from safetensors.torch import load_file

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token


class MyNet(nn.Module):

    target_neuron = None
    all_acts = []

    def __init__(self, model, input_ids, layer_number=None, use_sae=False):
        super().__init__()
        self.model = model

        self.input_ids = None
        if input_ids is not None:
            self.input_ids = input_ids[None, :]

        self.vision_tower = model.model.vision_tower
        self.projector = model.model.mm_projector

        self.llm = nn.Sequential()
        self.mlp = nn.Sequential()
        for i in range(layer_number+1):
            self.llm.append(model.model.layers[i])
        
        self.use_sae = use_sae
        if self.use_sae:
            self.map = nn.Linear(4096, 4096*8, bias=False)


    @torch.autocast(device_type="cuda")
    def forward(self, x):
        pos_ids = None
        attn_mask = None
        extract_pos = None
        if self.input_ids is not None:
            pos_ids = torch.arange(0, self.input_ids.shape[1], dtype=torch.long, device=self.input_ids.device)[None, :]
            _, pos_ids, attn_mask, _, x, _ = self.model.prepare_inputs_labels_for_multimodal(self.input_ids, pos_ids, None, None, None, x)
            extract_pos = (x.shape[1] // 2) + self.input_ids.shape[1]
        else:
            x = self.vision_tower(x)
            x = self.projector(x)
            pos_ids = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)[None, :]
            attn_mask = torch.ones((x.shape[0], 1, x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
            extract_pos = x.shape[1] // 2

        for l in self.llm:
            x = l(x, position_ids=pos_ids, attention_mask=attn_mask)[0]

        x = x[:, extract_pos, :]
        # x = x[:, -1, :]
        # x = torch.mean(x, dim=1)

        if self.use_sae:
            x = self.map(x)

        self.all_acts.append(x[0, self.target_neuron].detach().cpu().item())

        #   add empty spatial dimensions for lucent's activation extraction
        x = x[:, :, None, None]

        return x


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--basedir', type=str)
parser.add_argument('--module', type=str)
parser.add_argument('--neuron', type=int)
parser.add_argument('--sae', action='store_true')
parser.add_argument('--sae_root', type=str, required=False)
parser.add_argument('--type', type=str)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=16)
parser.add_argument('--steps', type=int, default=2560)
args = parser.parse_args()

device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

model_path = "liuhaotian/llava-v1.6-mistral-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device_str
)

# input_ids = tokenizer_image_token('What is this an image of?<image>', tokenizer, return_tensors='pt').to(device_str)

model = MyNet(model, None, int(args.module.split('.')[-1]), args.sae)
model.to(device_str).eval()

unit = None
if args.sae:
    sae_sparsity = load_file(os.path.join(args.sae_root, "sparsity.safetensors"))['sparsity']
    unit = torch.nonzero(sae_sparsity > -5)[args.neuron][0]
    sae_weights = load_file(os.path.join(args.sae_root, "sae_weights.safetensors"))
    states = model.state_dict()
    states['map.weight'] = torch.transpose(sae_weights['W_enc'], 0, 1)
    model.load_state_dict(states)
else:
    unit = args.neuron

# unit = args.neuron
savedir = os.path.join(args.basedir, args.network, args.module, f"unit{unit}")
Path(savedir).mkdir(parents=True, exist_ok=True)

model.target_neuron = unit

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
        torchvision.transforms.CenterCrop(336)
    ]
param_f = lambda: param.images.image(336, decorrelate=True)

print(f"BEGIN MODULE {args.module} NEURON {unit}")
obj = objectives.neuron('0', unit)

imgs = render.render_vis(nn.Sequential(model), obj, param_f=param_f, transforms=transforms, thresholds=(args.steps,), show_image=False)
img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
img.save(os.path.join(savedir, f"{args.steps}steps_distill_center.png"))

np.save(os.path.join(savedir, f"{args.steps}steps_distill_center_acts.npy"), np.array(model.all_acts))