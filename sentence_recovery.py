import argparse as ap
import random
import math
from pprint import pprint
import datetime as dt
from pathlib import Path
import itertools as it

import utils
import gboard2
import attack
import metrics

import tensorflow as tf
from anytree import Node
# setup gboard model
interpreter = gboard2.gboard_interpreter("gboard/gboard.tflite")
gboard_lstm = gboard2.create_gboard_lstm(interpreter)
gboard_embedding = gboard2.create_gboard_embedding(interpreter)
gboard_symbols = utils.load_symbols("gboard/gboard.syms")

def write_results(outf, original, extracted_words, gen_sentences, leven, f1):
    with outf.open('w') as f:
        for s in original:
            print(s, file=f)
        print("", file=f)
        print(*extracted_words, file=f)
        for s in gen_sentences:
            print(s, file=f)
        print("", file=f)
        print(f"{f1:.2f}, {leven:.2f}", file=f)
        
        


def main(args):
    sentences = list(utils.get_sentences(args.f))
    true_tokens = utils.get_true_tokens(gboard_symbols, sentences)
    dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences, batch_size=args.bs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.r)
    gboard_lstm.build(input_shape=input_shape)
    cutoff_str = "cutoff" if args.cutoff else "nocutoff"
    output_f = args.output_dir / f"sentence_exp_{args.f.name}_{args.e}_{args.bs}_{args.s}_{args.r}_{args.dp}_{cutoff_str}.results"
    gboard_lstm2 = gboard2.create_gboard_lstm(interpreter)
    attack.train_model(gboard_lstm2, dataset, args.e, optimizer,
                       dp_type=args.dp, noise_stddev=args.s)
    extracted_tokens = None
    if args.cutoff:
        extracted_tokens = attack.extract_best_negative_tokens(gboard_lstm2, gboard_lstm,
                                                                      true_tokens=true_tokens, stop=400)
    else:
        extracted_tokens, _ = attack.extract_negative_tokens(gboard_lstm2, gboard_lstm)

    extracted_tokens = extracted_tokens[0,:].numpy()

    prefixes = it.permutations(extracted_tokens, 1)
    generated_sentences = []
    for prefix in prefixes:
        prefix = [1] + list(prefix)                                                             
        prefix_embeddings = None                                                                
        for _ in range(3):                                                                      
            prefix_embeddings = utils.tokens2embeddings(gboard_embedding, tf.constant([prefix]))
            next_words = attack.next_words(prefix_embeddings, gboard_lstm2, extracted_tokens)   
            next_words.sort(key=lambda x: x[1], reverse=True)                                   
            prefix.append(next_words[0][0])                                                     
                                                                                                
        token_indices = [[0, i, t] for i, t in enumerate(prefix[1:])]                           
        pp0 = metrics.log_perplexity(prefix_embeddings, token_indices, gboard_lstm)             
        pp1 = metrics.log_perplexity(prefix_embeddings, token_indices, gboard_lstm2)            
        generated_sentences.append((prefix, ((pp0 - pp1) / pp0).numpy()))                    


    generated_sentences.sort(key=lambda x: x[1], reverse=True)
    generated_sentences = [(' '.join([utils.token2word(gboard_symbols, t) for t in s[0][1:]]), s[1]) for s in generated_sentences]
    print(f"generated {len(generated_sentences)} sentences")
    for s in generated_sentences:
        if s[1] < 0.5:
            break
        print(s)

    leven = metrics.calc_leven_actual(sentences, [s[0] for s in generated_sentences])
    f1 = metrics.f1(true_tokens, tf.expand_dims(extracted_tokens, axis=0)).numpy()
    print(f"f1 {f1}, leven {leven}")
    write_results(output_f, sentences, extracted_tokens, [f"{s[0]} {s[1]:.2f}" for s in generated_sentences],
                  leven, f1)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    output_dirname = f"sentence_exp_{str(dt.datetime.now()).replace(' ', '-')[:-7]}"
    parser.add_argument("-f", type=Path, required=True) # dataset file
    parser.add_argument('-e', type=int, required=True) # epochs
    parser.add_argument('-bs', type=int, required=True) # bs
    parser.add_argument('-s', type=float, required=True) # noise
    parser.add_argument('-r', type=float, default=1e-3) # learning rate
    parser.add_argument('-dp', type=float, default=1) # 1 = LOCAL_DP_SGD, otherwise = SINGLE_NOISE
    parser.add_argument('--cutoff', action='store_true', default=False)

    # default pl and pt generate 4 word sentences
    parser.add_argument('-pl', type=int, default=1) # initial prefix length
    parser.add_argument('-pt', type=int, default=3) # total generation length.

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
