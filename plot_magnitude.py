import argparse as ap
from pathlib import Path
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def get_params_from_file_name(filename):
    """ Assumes the filename to be of the format `f1_exp_file_epochs_bs_noise_lr_dptype.results'"""
    # remove the `f1_exp_' from the filname
    filename = filename.replace('mag_exp_', '')
    filename = filename.replace('.results', '')    
    nk, _, epochs, bs, noise, lr, dptype = tuple(filename.split('_'))
    return nk, epochs, bs, noise, lr, dptype 

def read_results(path):
    y = []
    with path.open('r') as f:
        for l in f:
            _y = l.strip()
            y.append(float(_y))

    return y


def main(args):
    colors = cm.rainbow(np.linspace(0, 1, len(args.results_files)))
    for color, path in zip(colors, args.results_files):
        _, epochs, _, noise, _, _ = get_params_from_file_name(path.name)
        label = None
        if args.group_by == "epochs":
            label = f"E = {epochs}"
        elif args.group_by == "noise":
            label = f"sigma = {noise}"

        y = read_results(path)
        plt.plot(y, color=color, label=label)
    plt.legend()
    plt.xlabel("words")
    plt.ylabel("magnitude")
    plt.title(args.title)
    plt.show()



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('results_files', type=Path, nargs='+')
    parser.add_argument('-t', '--title', type=str, default="")
    parser.add_argument('--group-by', type=str, default=None)
    args = parser.parse_args()
    main(args)
