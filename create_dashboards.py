import argparse
import sys
import os
import warnings
warnings.simplefilter("ignore")
import numpy as np
import torch
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_lens.load_model import load_model

sys.path.append('LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

sys.path.append('SAEDashboard')
from SAEDashboard.sae_dashboard.sae_vis_data import SaeVisConfig
from SAEDashboard.sae_dashboard.sae_vis_runner import SaeVisRunner
from SAEDashboard.sae_dashboard.data_writing_fns import save_feature_centric_vis

from hook_llava import load_hooked_llava


parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int)
parser.add_argument('--sae_id', type=int)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

device_str = f'cuda:{args.device}'
use_llava = True
dashboard_root = '/media/andrelongon/DATA/visual_llm_out/resids/dashboards'

mean_resids = torch.tensor(np.load(f'/media/andrelongon/DATA/visual_llm_out/resids/layer{args.layer}_mean_mlp_resid_out.npy'))
mean_streams = torch.tensor(np.load(f'/media/andrelongon/DATA/visual_llm_out/resids/layer{args.sae_id}_mean_txt_pre_out.npy'))
resids_norms = torch.linalg.vector_norm(mean_resids, dim=-1)
streams_norms = torch.linalg.vector_norm(mean_streams, dim=-1)

norm_ratios = streams_norms / (resids_norms + 1e-10)
mean_resids = torch.mul(mean_resids, norm_ratios[:, None].expand(-1, mean_resids.shape[1]))

#   Obtain mean residual vectors, pass to pretrained SAE and obtain topk activating latents.  Generate dashboards for those latents.
sae, _, sparsity = SAE.from_pretrained(
    release="mistral-7b-res-wg",
    sae_id=f"blocks.{args.sae_id}.hook_resid_pre",
    device=device_str
)

latents = sae.encode_standard(torch.mean(mean_resids, dim=0).to(device_str))
# latents = sae.encode_standard(mean_resids[0].to(device_str))
_, top_latents = torch.topk(latents, 16)

model = None
if use_llava:
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device='cpu'
    )
    model = load_hooked_llava('/media/andrelongon/DATA/visual_llm_out/llava_mistral_lens.pth', tokenizer=tokenizer, device='cpu')
    model = model.to(device_str)
else:
    model = load_model(
        'HookedTransformer',
        'mistral-7b',
        device=device_str
    )

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)
token_dataset = tokenize_and_concatenate(
    dataset=dataset,
    tokenizer=model.tokenizer,
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)
feature_vis_config = SaeVisConfig(
    hook_point=f"blocks.{args.sae_id}.hook_resid_pre",
    features=top_latents.tolist(),
    minibatch_size_features=64,
    minibatch_size_tokens=128,
    # quantile_feature_batch_size=64,
    verbose=True,
    device=device_str,
)
viz_data = SaeVisRunner(feature_vis_config).run(
    encoder=sae,
    model=model,
    tokens=token_dataset[:10000]["tokens"].to(device_str)
)

json_path = os.path.join(dashboard_root, f"scaled_mean_resid{args.layer}llava_block{args.sae_id}_top16_latents.json")
viz_data.save_json(json_path)

filename = os.path.join(dashboard_root, f"scaled_mean_resid{args.layer}_llava_block{args.sae_id}_top16_latents.html")
save_feature_centric_vis(sae_vis_data=viz_data, filename=filename)