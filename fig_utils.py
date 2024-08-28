import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from torchvision.transforms import ToPILImage
from torchvision.io import read_image
from torchvision.utils import make_grid


def plot_fz_steps(fz_root, features, interval):
    legend_entries = []
    for f in features:
        acts = np.load(os.path.join(fz_root, f'unit{f}', f'2560steps_distill_center_acts.npy'))
        idxs = list(range(0, acts.shape[0], interval))

        if acts.shape[0]-1 not in idxs:
            idxs.append(acts.shape[0]-1)

        plt.plot(idxs, acts[idxs])
        legend_entries.append(f'Feature {f}')

    plt.title(f'Feature Activations during Feature Viz\nLayer {fz_root.split('/')[-1]}')
    plt.xlabel('Optimization Step')
    plt.ylabel('Activation')
    plt.legend(legend_entries, loc='upper left')

    plt.savefig(os.path.join(fz_root, 'acts_plot.png'))


#   TODO:  computer ANOVA and print F-score and associated p-value.  Do the number of features need to be identical?
def plot_score_histo(results_root, num_feats):
    llava_file = open(os.path.join(results_root, 'llava_100feats_autointerp_results.json'))
    llava_results = json.load(llava_file)
    llava_file = open(os.path.join(results_root, 'llava_100feats_batch2_autointerp_results.json'))
    llava_results += json.load(llava_file)
    # llava_file = open(os.path.join(results_root, 'llava_100feats_batch3_autointerp_results.json'))
    # llava_results += json.load(llava_file)
    llava_file = open(os.path.join(results_root, 'llava_100feats_batch4_autointerp_results.json'))
    llava_results += json.load(llava_file)
    llava_results = np.array([result['score'] for result in llava_results[:num_feats]])
    print(llava_results.shape)
    print(f"Llava Mean: {np.mean(llava_results)}")

    mistral_file = open(os.path.join(results_root, 'mistral_100feats_autointerp_results.json'))
    mistral_results = json.load(mistral_file)
    mistral_file = open(os.path.join(results_root, 'mistral_100feats_batch2_autointerp_results.json'))
    mistral_results += json.load(mistral_file)
    mistral_file = open(os.path.join(results_root, 'mistral_100feats_batch3_autointerp_results.json'))
    mistral_results += json.load(mistral_file)
    mistral_file = open(os.path.join(results_root, 'mistral_100feats_batch3_autointerp_results.json'))
    mistral_results += json.load(mistral_file)
    mistral_results = np.array([result['score'] for result in mistral_results])
    print(mistral_results.shape)
    print(f"Mistral Mean: {np.mean(mistral_results)}")

    anova_result = f_oneway(llava_results, mistral_results)
    print(anova_result)

    plt.hist((llava_results, mistral_results), range=(-1, 1))
    plt.title(f'Auto-Interp Scores (gpt-4o-mini) for SAE Features in {results_root.split('/')[-1]}')
    plt.legend(['Llava', 'Mistral'], loc='upper left')
    plt.xlabel('Score')
    plt.savefig(os.path.join(results_root, 'histogram.png'))


def make_fz_grid(fz_root, features):
    imgs = []
    for f in features:
        img = read_image(os.path.join(fz_root, f'unit{f}', '2560steps_distill_center.png'))
        imgs.append(img)

    grid = make_grid(imgs, nrow=3, padding=8)
    grid = ToPILImage()(grid)

    grid.save(os.path.join(fz_root, f"fz_grid.png"))




# fz_root = '/media/andrelongon/DATA/visual_llm_study/feature_viz/neuron/llava/blocks.16'
# features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# plot_fz_steps(fz_root, features, 100)
# make_fz_grid(fz_root, features)

# plot_score_histo('/media/andrelongon/DATA/visual_llm_study/autointerp_results/blocks.8', 248)
plot_score_histo('/media/andrelongon/DATA/visual_llm_study/autointerp_results/blocks.16', 248)