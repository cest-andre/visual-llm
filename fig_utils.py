import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from torchvision.transforms import ToPILImage
from torchvision.io import read_image
from torchvision.utils import make_grid


def plot_score_histo(results_root, num_feats, interval=64):
    block_num = results_root.split('/')[-1]

    mistral_base_results = []
    mistral_it_results = []
    llava_results = []
    for i in range(0, 256, interval):
        file = open(os.path.join(results_root, f'mistral_base_feats{i}-{i+interval}_autointerp_results.json'))
        mistral_base_results += json.load(file)

        file = open(os.path.join(results_root, f'mistral_it_feats{i}-{i+interval}_autointerp_results.json'))
        mistral_it_results += json.load(file)

        file = open(os.path.join(results_root, f'llava_feats{i}-{i+interval}_autointerp_results.json'))
        llava_results += json.load(file)

    print(len(mistral_base_results))
    mistral_base_results = np.array([result['score'] for result in mistral_base_results[:num_feats]])
    print(f"Mistral base Mean: {np.mean(mistral_base_results)}")

    print(len(mistral_it_results))
    mistral_it_results = np.array([result['score'] for result in mistral_it_results[:num_feats]])
    print(f"Mistral-IT Mean: {np.mean(mistral_it_results)}")

    print(len(llava_results))
    llava_results = np.array([result['score'] for result in llava_results[:num_feats]])
    print(f"Llava Mean: {np.mean(llava_results)}")

    anova_result = f_oneway(mistral_base_results, mistral_it_results)
    print(f'Mistral Base-IT ANOVA:  {anova_result}')

    anova_result = f_oneway(mistral_base_results, llava_results)
    print(f'Mistral Base-Llava ANOVA:  {anova_result}')

    anova_result = f_oneway(mistral_it_results, llava_results)
    print(f'Mistral IT-Llava ANOVA:  {anova_result}')

    plt.hist((mistral_base_results, mistral_it_results, llava_results), range=(-1, 1))
    plt.title(f'Auto-Interp Scores (gpt-4o-mini) for SAE Features in {block_num}')
    plt.legend(['Mistral Base', 'Mistral-IT', 'Llava'], loc='upper left')
    plt.xlabel('Score')
    plt.savefig(os.path.join(results_root, f'{block_num}_all_histogram.png'))
    plt.close()


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


def make_fz_grid(fz_root, features):
    imgs = []
    for f in features:
        img = read_image(os.path.join(fz_root, f'unit{f}', '2560steps_distill_center.png'))
        imgs.append(img)

    grid = make_grid(imgs, nrow=3, padding=8)
    grid = ToPILImage()(grid)

    grid.save(os.path.join(fz_root, f"fz_grid.png"))


plot_score_histo('/media/andrelongon/DATA/DO_NOT_DELETE/autointerp_results/blocks.16', 248)