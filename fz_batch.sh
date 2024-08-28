# !/bin/bash

for i in {3..8..1}
do
    python feature_viz.py --network llava --module blocks.16 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/neuron --device 1 --jitter 24 --neuron $i
    # python feature_viz.py --network llava --module final-blocks.8 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.8/p48gd7yj/final_102400000 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/sae_feature --device 0 --jitter 24 --neuron $i
    # python feature_viz.py --network llava --module final-blocks.16 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.16/siwnxuym/final_102400000 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/sae_feature --device 1 --jitter 24 --neuron $i
    # python feature_viz.py --network llava --module blocks.16 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.16/f0sttp3g/61444096 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/sae_feature --device 1 --jitter 24 --neuron $i
    # python feature_viz.py --network llava --module blocks.24 --sae --sae_root /media/andrelongon/DATA/visual_llm_study/sae_checkpoints/llava/blocks.16/f0sttp3g/61444096 --basedir /media/andrelongon/DATA/visual_llm_study/feature_viz/neuron --device 1 --jitter 24 --neuron $i
done