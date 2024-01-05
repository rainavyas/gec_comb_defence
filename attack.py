'''
Perform greedy attack or evaluation
'''

import sys
import os
import json

from src.tools.args import core_args, attack_args
from src.tools.tools import set_seeds
from src.data.load_data import load_data
from src.inference.model_selector import select_model
from src.attack.attacker import GreedyAttacker

if __name__ == "__main__":

    # get command line arguments
    args, c = core_args()
    aargs, a = attack_args()
    print(args)
    print(aargs)

    set_seeds(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # load model
    model = select_model(args)

    if aargs.eval:
        '''
        Find next best attack word from cached files
        Evaluate the attack phrase wrt to fraction of samples of w/ 0 edits
        '''
        # data
        data, _ = load_data(args.data_name)

        # get next best word
        curr_adv_phrase = aargs.prev_phrase
        pos = len(curr_adv_phrase.split(' '))+1 if curr_adv_phrase != '' else 1
        base_path = f'{aargs.base_path}/pos{pos}'
        word, train_score = GreedyAttacker.next_best_word(base_path)
        print('Next best word')
        print(word, train_score)
        print()

        # evaluate fraction of samples with 0 edits
        attack_phrase = curr_adv_phrase + ' ' + word
        frac = GreedyAttacker._eval_attack(model, data, attack_phrase)
        print('Evaluating attack phrase:', attack_phrase)
        print('Evaluating on ', args.data_name)
        print('Fraction of samples with 0 edits: ', frac)

    else:
        # perform next iteration of greedy attack (train)

        # save / load test words
        fpath = 'experiments/words.txt'

        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                word_list = json.load(f)

        else:
            import nltk
            nltk.download('words')
            from nltk.corpus import words
            word_list = words.words()
            word_list = list(set(word_list))[:50000]

            with open(fpath, 'w') as f:
                json.dump(word_list, f)

        # select vocab segment if array job
        if aargs.array_job_id != -1:
            start = aargs.array_job_id*aargs.array_word_size
            end = start+aargs.array_word_size
            word_list = word_list[start:end]
        
        # load data
        data, _ = load_data(aargs.train_data_name)
        
        # perform next iteration of attack
        attacker = GreedyAttacker(model, word_list)
        _, _ = attacker.attack_next_word(data, aargs.base_path, curr_adv_phrase=aargs.prev_phrase, array_job_id=aargs.array_job_id)