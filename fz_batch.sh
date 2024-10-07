# !/bin/bash

# python feature_viz.py --network llava --module blocks.8 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/neuron --device 0 --jitter 24 --neuron $i
# python feature_viz.py --network llava --module blocks.16 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/neuron --device 0 --jitter 24 --neuron $i
# python feature_viz.py --network llava --module final-blocks.8 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.8/p48gd7yj/final_102400000 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/sae_feature --device 0 --jitter 24
# python feature_viz.py --network llava --module final-blocks.16 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.16/siwnxuym/final_102400000 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/sae_feature --device 0 --jitter 24


# python generate_explanations.py --network mistral_base --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 0 --stop_idx 64
# python generate_explanations.py --network mistral_base --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 64 --stop_idx 128
# python generate_explanations.py --network mistral_base --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 128 --stop_idx 192
# python generate_explanations.py --network mistral_base --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 192 --stop_idx 256

# python generate_explanations.py --network mistral_it --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 0 --stop_idx 64
# python generate_explanations.py --network mistral_it --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 64 --stop_idx 128
# python generate_explanations.py --network mistral_it --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 128 --stop_idx 192
# python generate_explanations.py --network mistral_it --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 192 --stop_idx 256

# python generate_explanations.py --network llava --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 0 --stop_idx 64
# python generate_explanations.py --network llava --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 64 --stop_idx 128
# python generate_explanations.py --network llava --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 128 --stop_idx 192
# python generate_explanations.py --network llava --layer blocks.8 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 192 --stop_idx 256




# python generate_explanations.py --network llava --layer blocks.16 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 0 --stop_idx 64
# python generate_explanations.py --network llava --layer blocks.16 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 64 --stop_idx 128
# python generate_explanations.py --network llava --layer blocks.16 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 128 --stop_idx 192
# python generate_explanations.py --network llava --layer blocks.16 --basedir /media/andrelongon/DATA/DO_NOT_DELETE --sae_ckpt NA --start_idx 192 --stop_idx 256