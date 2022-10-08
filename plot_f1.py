"""This program reads a file which contains lines of the form epoch,
f1

"""
import argparse as ap
from pathlib import Path
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np




def read_data(paths, group_by):
    """ For when we are plotting single epoch points not lines """
    result = {}
    for path in paths:
        nk, epochs, bs, noise, lr, dptype, cutoff = get_params_from_file_name(path.name)
        x, y = read_results(path) # epoch, f1
        key = None
        if group_by == "nk" :
            key = nk
        elif group_by == "epochs":
            key = epochs

        if key not in result:
            result[key] = {}
        result[key][noise] = y
    return result
            
def get_params_from_file_name(filename):
    """Assumes the filename to be of the format
    `f1_exp_file_epochs_bs_noise_lr_dptype_cutofftype.results'"""
    # remove the `f1_exp_' from the filname
    filename = filename.replace('f1_exp_', '')
    filename = filename.replace('.results', '')    
    nk, _, epochs, bs, noise, lr, dptype, cutoff = tuple(filename.split('_'))
    cutoff = True if cutoff == 'cutoff' else False
    return nk, epochs, bs, noise, lr, dptype, cutoff

def read_results(path):
    x = []; y = []
    with path.open('r') as f:
        for l in f:
            _x, _y = l.strip().split()
            x.append(int(_x)); y.append(float(_y))

    return x, y
            
def main(args):
    if not args.single:
        colors = cm.rainbow(np.linspace(0, 1, len(args.results_files)))
        for color, path in zip(colors, args.results_files):
            nk, _, _, noise, _, _, _ = get_params_from_file_name(path.name)
            x, y = read_results(path)
            print(nk)
            pprint((x,y))
            plt.plot(x, y, color=color, label=f"{args.group_by if args.group_by else 'sigma'} = {nk if args.group_by == 'nk' else noise}")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("f1")
        plt.title(args.title)
        plt.xscale('log')
        plt.ylim([0,1])
        plt.show()
        return

    # single points
    data = read_data(args.results_files, args.group_by)
    pprint(data)
    colors = cm.rainbow(np.linspace(0, 1, 5))
    should_label = True
    for i, g in enumerate(sorted(data.keys(), key=lambda x: int(x))):
        for j, noise in enumerate(data[g]):
            label = f"sigma = {noise}" if should_label else None
            plt.plot(g, data[g][noise], color=colors[j], label=label,
                     marker='+', markersize=15, linewidth=1)
        if i == 0:
            should_label = False
    plt.legend()
    plt.xlabel(args.group_by)
    plt.ylabel("f1")
    plt.title(args.title)
    plt.ylim([0,1])
    plt.show()
    return

    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('results_files', type=Path, nargs='+')
    parser.add_argument('-t', '--title', type=str, default="")
    # if we are plotting the results for f1 when noise is added to the
    # final model parameters. In this case it is just 1 datapoint
    # because we aren't plotting it over all the epochs, just after
    # the final one, where the noise is added. This also useful when
    # plotting fedsgd results since there is only one epoch
    parser.add_argument('--single', action="store_true", default=False)
    parser.add_argument('--group-by', type=str, default=None)
    args = parser.parse_args()
    main(args)
