import os
import asyncio
import json
import math
import torch
import sys
import time
import argparse
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.load_model import load_model
from sae_lens.analysis.neuronpedia_integration import simulate_and_score
from transformer_lens.utils import tokenize_and_concatenate

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.explainer import (
    HARMONY_V4_MODELS,
    ContextSize,
    TokenActivationPairExplainer,
)
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.simulator import LogprobFreeExplanationTokenSimulator

sys.path.append('SAEDashboard')
from SAEDashboard.sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from SAEDashboard.sae_dashboard.sae_vis_runner import SaeVisRunner
from SAEDashboard.sae_dashboard.data_writing_fns import save_feature_centric_vis

from hook_llava import load_hooked_llava


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--layer', type=str)
parser.add_argument('--basedir', type=str)
parser.add_argument('--sae_ckpt', type=str)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--stop_idx', type=int, default=100)
args = parser.parse_args()

network = args.network
layer = args.layer
basedir = args.basedir
autointerp_root = os.path.join(basedir, "autointerp_results", layer)
dashboard_root = os.path.join(basedir, f'sae_dashboard_viz/{network}/{layer}')
sae_root = os.path.join(basedir, f'sae_checkpoints/{network}/{layer}/{args.sae_ckpt}/final_102400000')


def save_feature_dashboard(start_idx, stop_idx, llava=True, save_html=True, device='cuda:1'):
    if llava:
        model = load_hooked_llava(states_path=os.path.join(basedir, f'{network}_lens_weights.pth'), device=device)

        sae = SAE.load_from_pretrained(
            sae_root, device=device, dtype='float32'
        )
        sparsity = load_file(os.path.join(sae_root, "sparsity.safetensors"))['sparsity']
    else:
        model = load_model(
            'HookedTransformer',
            'mistral-7b',
            device=device
        )

        sae, _, sparsity = SAE.from_pretrained(
            release = "mistral-7b-res-wg",
            sae_id = f"{layer}.hook_resid_pre",
            device = device
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

    if llava:
        feat_idx = torch.squeeze(torch.nonzero(sparsity > -5)).tolist()[start_idx:stop_idx]
    else:
        feat_idx = list(range(start_idx, stop_idx))

    feature_vis_config = SaeVisConfig(
        hook_point=f"{layer}.hook_resid_pre",
        features=feat_idx,
        minibatch_size_features=64,
        minibatch_size_tokens=128,
        # quantile_feature_batch_size=64,
        verbose=True,
        device=device,
    )

    viz_data = SaeVisRunner(feature_vis_config).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:10000]["tokens"].to(device)
    )

    json_path = os.path.join(dashboard_root, f"{network}_feats{args.start_idx}-{args.stop_idx}_25ksubset_pile-10k.json")
    viz_data.save_json(json_path)

    if save_html:
        filename = os.path.join(dashboard_root, f"{network}_feats{args.start_idx}-{args.stop_idx}_25ksubset_pile-10k.html")
        save_feature_centric_vis(sae_vis_data=viz_data, filename=filename)

    return json_path


def explain_features(json_path):
    model = load_hooked_llava(states_path=os.path.join(basedir, f'{network}_lens_weights.pth'), device='cpu')

    HARMONY_V4_MODELS.append('gpt-4o-mini')

    #   Using 4o mini for both due to lower cost
    explainer_model_name = 'gpt-4o-mini'
    scorer_model_name = 'gpt-4o-mini'

    explainer = TokenActivationPairExplainer(
        model_name='gpt-4o-mini',
        prompt_format=PromptFormat.HARMONY_V4,
        context_size=ContextSize.SIXTEEN_K,
        max_concurrent=1,
    )

    results = []
    total_score = 0
    with open(json_path) as db_file:
        data = json.load(db_file)['feature_data_dict']

        for f in data.keys():
            result = {}
            act_groups = data[f]['sequence_data']['seq_group_data']

            act_records = []
            for act in act_groups[0]['seq_data']:
                tokens = model.tokenizer.batch_decode([[t] for t in act['token_ids']])
                act_records.append(ActivationRecord(tokens=tokens, activations=act['feat_acts']))

            explanation = asyncio.run(explainer.generate_explanations(
                all_activation_records=act_records,
                max_activation=calculate_max_activation(act_records),
                num_samples=1,
                temperature=0
            ))

            assert len(explanation) == 1

            result['feature'] = f
            result['explanation'] = explanation[0].rstrip(".")

            temp_activation_records = [
                ActivationRecord(
                    tokens=[
                        token.replace("<|endoftext|>", "<|not_endoftext|>")
                        .replace(" 55", "_55")
                        .encode("ascii", errors="backslashreplace")
                        .decode("ascii")
                        for token in record.tokens
                    ],
                    activations=record.activations,
                )
                for record in act_records
            ]

            simulator = UncalibratedNeuronSimulator(
                LogprobFreeExplanationTokenSimulator(
                    scorer_model_name,
                    explanation,
                    json_mode=True,
                    max_concurrent=20,
                    few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
                    prompt_format=PromptFormat.HARMONY_V4,
                )
            )
            scored_simulation = asyncio.run(simulate_and_score(
                simulator, temp_activation_records
            ))

            time.sleep(5)   # wait to avoid openai rate_exceed error

            score = scored_simulation.get_preferred_score()
            
            if not math.isnan(score):
                total_score += score
                result['score'] = score
                results.append(result)
            else:
                print(f"NAN SCORE.  SKIPPING FEATURE {f}")

    print(f"Mean score: {total_score / len(results)}")

    with open(os.path.join(autointerp_root, f"{network}_feats{args.start_idx}-{args.stop_idx}_autointerp_results.json"), 'w') as f:
        json.dump(results, f)


def extract_top_features(json_path):
    with open(json_path) as interp_file:
        interps = json.load(interp_file)

        quotes = 0
        for interp in interps:
            if interp['score'] >= 0.5 and ("'" in interp['explanation'] or '"' in interp['explanation']):
                print(interp)
                quotes += 1

        print(f'Quoted interp percent:  {quotes / len(interps)}')


json_path = save_feature_dashboard(args.start_idx, args.stop_idx, llava=False)
# json_path = os.path.join(dashboard_root, f"{network}_feats{args.start_idx}-{args.stop_idx}_25ksubset_pile-10k.json")
# explain_features(json_path)

# print(network)
# json_path = os.path.join(autointerp_root, f"{network}_feats{args.start_idx}-{args.stop_idx}_autointerp_results.json")
# extract_top_features(json_path)