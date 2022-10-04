import argparse as ap
from pathlib import Path
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# { nk : [(f1, leven) ...] }



def get_params_from_file_name(filename):
    """Assumes the filename to be of the format
    `f1_exp_file_epochs_bs_noise_lr_dptype_cutofftype.results'"""
    # remove the `f1_exp_' from the filname
    filename = filename.replace('sentence_exp_', '')
    filename = filename.replace('.results', '')    
    nk, _, epochs, bs, noise, lr, dptype, cutoff = tuple(filename.split('_'))
    cutoff = True if cutoff == 'cutoff' else False
    return nk, epochs, bs, noise, lr, dptype, cutoff

def read_results(path):
    """ reads the final line (f1, leven) """
    with path.open('r') as f:
        f1, leven = tuple(map(float, f.readlines()[-1].strip().split(', ')))
        print(f1, leven)

    return f1, leven




def main(args):
    results = {}
    for path in args.results_files:
        nk, _, _, _, _, _, _ = get_params_from_file_name(path.name)
        metrics = read_results(path)
        if args.group_by == "nk":
            if nk in results:
                results[nk].append(metrics)
            else:
                results[nk] = [metrics]
        else:
            raise NotImplemented
        
    pprint(results)
    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    for c, nk in zip(colors, results):
        x = [r[0] for r in results[nk]]
        y = [r[1] for r in results[nk]]
        plt.scatter(x, y, color=c, label=f"nk = {nk}")

    plt.xlabel("f1")
    plt.ylabel("levenshtein ratio")
    plt.ylim([0,100])
    plt.xlim([0,1])
    plt.legend()
    plt.title(args.title)
    plt.show()



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('results_files', type=Path, nargs='+')
    parser.add_argument('--group-by', type=str, default="nk")
    parser.add_argument('-t', '--title', type=str, default="")
    args = parser.parse_args()
    main(args)
