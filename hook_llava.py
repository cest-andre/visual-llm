import torch
from transformers import AutoModelForCausalLM
from sae_lens.load_model import load_model
import transformer_lens.loading_from_pretrained as loading
from transformer_lens.pretrained.weight_conversions.mistral import convert_mistral_weights

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def load_hooked_llava(states_path=None, device='cuda'):
    llava_states = None
    if states_path is None:
        llava_states = extract_llava_llm_weights()
    else:
        llava_states = torch.load(states_path)

    hooked_model = load_model(
        'HookedTransformer',
        'mistral-7b-instruct',
        device=device
    )
    hooked_model.load_and_process_state_dict(llava_states)

    return hooked_model


def convert_llava_lens_weights(save_path=None):
    model_path = "liuhaotian/llava-v1.6-mistral-7b"

    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    llava_states = llava_model.state_dict()
    states = base_model.state_dict()

    for k in llava_states.keys():
        if k in states.keys():
            states[k] = llava_states[k]

    base_model.load_state_dict(states)

    cfg = loading.get_pretrained_model_config(
        'mistralai/Mistral-7B-Instruct-v0.1',
        hf_cfg={},
        checkpoint_index=None,
        checkpoint_value=None,
        fold_ln=True,
        device=None,
        n_devices=1,
        default_prepend_bos=True,
        dtype=torch.float32,
    )
    states = convert_mistral_weights(base_model, cfg)

    if save_path is not None:
        torch.save(states, save_path)

    return states


def convert_mistral_v2_weights(save_path=None):
    base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    cfg = loading.get_pretrained_model_config(
        'mistralai/Mistral-7B-Instruct-v0.1',
        hf_cfg={},
        checkpoint_index=None,
        checkpoint_value=None,
        fold_ln=True,
        device=None,
        n_devices=1,
        default_prepend_bos=True,
        dtype=torch.float32,
    )
    states = convert_mistral_weights(base_model, cfg)

    if save_path is not None:
        torch.save(states, save_path)

    return states