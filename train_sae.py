import torch
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.load_model import load_model

from hook_llava import load_hooked_llava


total_training_steps = 25000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = total_training_steps // 10
lr_decay_steps = total_training_steps // 10
l1_warm_up_steps = total_training_steps // 20

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="mistral-7b-instruct",
    hook_name="blocks.16.hook_resid_pre",
    hook_layer=16,
    d_in=4096,
    dataset_path="monology/pile-uncopyrighted",
    prepend_bos=False,
    is_dataset_tokenized=False,
    streaming=True,
    # SAE Parameters
    architecture='gated',
    mse_loss_normalization=None,
    expansion_factor=8,
    b_dec_init_method="geometric_median",
    apply_b_dec_to_input=False,
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="none",
    # Training Parameters
    lr=2e-4,
    lr_end=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="cosineannealing",
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l1_coefficient=0.012,
    l1_warm_up_steps=l1_warm_up_steps,
    lp_norm=0.5,
    train_batch_size_tokens=batch_size,
    context_size=256,
    # Activation Store Parameters
    n_batches_in_buffer=16,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=8,
    act_store_device='cuda:1',
    # Resampling protocol
    use_ghost_grads=False,
    feature_sampling_window=2000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-8,
    # WANDB
    log_to_wandb=True,
    wandb_project="sae_lens_llava",
    wandb_log_frequency=10,
    eval_every_n_wandb_logs=100,
    # Misc
    device="cuda:1",
    seed=42,
    n_checkpoints=5,
    checkpoint_path="/media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.16",
    dtype="float32"
)

model = load_hooked_llava(states_path='/media/andrelongon/DATA/visual_llm_study/mistral-v0.2_lens_weights.pth', device='cuda:1')
sparse_autoencoder = SAETrainingRunner(cfg, override_model=model).run()