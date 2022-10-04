import argparse as ap
import datetime as dt
from pathlib import Path
from pprint import pprint


import utils
import gboard2
import attack
import metrics

import tensorflow as tf


GBOARD_MODEL_FILE = "gboard/gboard.tflite"
GBOARD_SYMBOLS_FILE = "gboard/gboard.syms"

def main(args):
    interpreter = gboard2.gboard_interpreter(GBOARD_MODEL_FILE)
    gboard_lstm = gboard2.create_gboard_lstm(interpreter)
    gboard_embedding = gboard2.create_gboard_embedding(interpreter)
    gboard_symbols = utils.load_symbols(GBOARD_SYMBOLS_FILE)    

    sentences = list(utils.get_sentences(args.f))
    true_tokens = utils.get_true_tokens(gboard_symbols, sentences)
    dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences,
                                                   batch_size=args.bs)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.r)
    gboard_lstm.build(input_shape=input_shape)

    cutoff_str = "cutoff" if args.cutoff else "nocutoff"
    output_f = args.output_dir / f"f1_exp_{args.f.name}_{args.e}_{args.bs}_{args.s}_{args.r}_{args.dp}_{cutoff_str}.results"

    f1s = []

    def after_epoch(e, lstm):
        # calculate f1 at this epoch and save it
        recovered_tokens = None
        if args.cutoff:
            recovered_tokens = attack.extract_best_negative_tokens(lstm, gboard_lstm,
                                                                      true_tokens=true_tokens, stop=400)
        else:
            recovered_tokens, _ = attack.extract_negative_tokens(lstm, gboard_lstm)
        f1 = metrics.f1(true_tokens, recovered_tokens)
        print(f"recovered {recovered_tokens.shape} tokens (f1: {f1:.4f})")
        if not args.measure_final_only:
            f1s.append((e,f1.numpy()))
        elif e == args.e - 1 and args.measure_final_only:
            f1s.append((e,f1.numpy()))


    gboard_lstm2 = gboard2.create_gboard_lstm(interpreter)

    attack.train_model(gboard_lstm2, dataset, args.e, optimizer,
                       dp_type=args.dp, noise_stddev=args.s,
                       after_epoch_func=after_epoch)
    # pprint(f1s)

    utils.output_results(output_f, f1s)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    output_dirname = f"f1_exp_{str(dt.datetime.now()).replace(' ', '-')[:-7]}"
    parser.add_argument("-f", type=Path, required=True) # dataset file
    parser.add_argument('-e', type=int, required=True) # epochs
    parser.add_argument('-bs', type=int, required=True) # bs
    parser.add_argument('-s', type=float, required=True) # noise
    parser.add_argument('-r', type=float, default=1e-3) # learning rate
    parser.add_argument('-dp', type=float, default=1) # 1 = LOCAL_DP_SGD, otherwise = SINGLE_NOISE
    parser.add_argument('--cutoff', action='store_true', default=False)
    parser.add_argument('--measure-final-only', action='store_true', default=False) # whether to measure the f1 after the final epoch only
    parser.add_argument("-o", "--output-dir", type=Path, default=output_dirname)    
    args = parser.parse_args()


    if args.output_dir and not args.output_dir.exists():
        print(f'--output-dir {args.output_dir} does not exist, creating it...')
        args.output_dir.mkdir(parents=True)

    if args.s == 0:
        args.dp = None
    elif args.dp == 1:
        args.dp = attack.LOCAL_DP_SGD
    else:
        args.dp = attack.SINGLE_NOISE

    main(args)
