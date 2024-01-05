import os
import json
from tqdm import tqdm

from src.tools.saving import next_dir

class GreedyAttacker:
    def __init__(self, model, word_list):
        self.word_list = word_list
        self.model = model
    
    def attack_next_word(self, sentences, cache_path, curr_adv_phrase='', array_job_id=-1):
        # check for cache
        pos = len(curr_adv_phrase.split(' '))+1 if curr_adv_phrase != '' else 1
        path = next_dir(cache_path, f'pos{pos}')
        if array_job_id != -1:
            path = next_dir(path, f'array_job{array_job_id}')

        fpath_prev = f'{path}/prev.txt'
        fpath_scores = f'{path}/scores.txt'
        if os.path.isfile(fpath_prev):
            with open(fpath_prev, 'r') as f:
                prev = json.load(f)
            with open(fpath_scores, 'r') as f:
                word_2_score = json.load(f)

            return prev, word_2_score
        
        prev_score = self._avg_score(sentences, curr_adv_phrase)
        word_2_score = {}
        for word in tqdm(self.word_list):
            if curr_adv_phrase == '':
                adv_phrase = word
            else:
                adv_phrase = curr_adv_phrase + ' ' + word
            score = self._avg_score(sentences, adv_phrase)
            word_2_score[word] = score

        # cache
        with open(fpath_prev, 'w') as f:
            prev = {'prev-adv-phrase': curr_adv_phrase, 'score':prev_score}
            json.dump(prev, f)
        with open(fpath_scores, 'w') as f:
            json.dump(word_2_score, f)
        
        return prev, word_2_score
    
    def _avg_score(self, sentences, attack_phrase=''):
        '''
        Returns the avg number of edis across the sentences
        '''
        edit_counts = 0
        for sent in sentences:
            sent_attack = self._prep_attacked(sent, attack_phrase)
            _, cnt = self.model.predict([sent_attack], return_cnt=True)
            edit_counts += cnt
        return edit_counts/len(sentences)

    @classmethod
    def _eval_attack(cls, model, sentences, attack_phrase):
        '''
        Returns the fraction of samples with 0 edits
        '''
        count = 0
        for sent in sentences:
            sent_attack = cls._prep_attacked(sent, attack_phrase)
            _, cnt = model.predict([sent_attack], return_cnt=True)
            if cnt == 0:
                count +=1
        return count/len(sentences)

    @staticmethod
    def _prep_attacked(sentence, attack_phrase):
        if attack_phrase == '':
            sent_attack = sentence
        else:
            if sentence[-1] == '.' or sentence[-1] == ',':
                sent_attack = sentence[:-1] + ' ' + attack_phrase + ' .'
            else:
                sent_attack = sentence + ' ' + attack_phrase + ' .'
        return sent_attack

    @staticmethod
    def next_best_word(base_path, pos=1):
        '''
            base_path: directory with scores.txt and prev.txt (or array_job files)
            Give the next best (lowest avg edits) word from output saved files
        '''

        def best_from_dict(word_2_score, pos=1):
            prev = [None, 1000]
            best = [None, 1000]

            for k,v in word_2_score.items():
                if v<best[1]:
                    prev[0] = best[0]
                    prev[1] = best[1]
                    best[0]=k
                    best[1]=v
                elif v<prev[1]:
                    prev[0]=k
                    prev[1]=v
            if pos==1:
                return best[0], best[1]
            elif pos==2:
                return prev[0], prev[1]
            else:
                print("Not supported pos")

        if os.path.isfile(f'{base_path}/scores.txt'):
            with open(f'{base_path}/scores.txt', 'r') as f:
                word_2_score = json.load(f)
            return best_from_dict(word_2_score, pos=pos)
        
        elif os.path.isdir(f'{base_path}/array_job10'):
            combined = {}
            for i in range(230):
                try:
                    with open(f'{base_path}/array_job{i}/scores.txt', 'r') as f:
                        word_2_score = json.load(f)
                except:
                    continue
                combined = {**combined, **word_2_score}
            
            return best_from_dict(combined, pos=pos)

        else:
            raise ValueError("No cached scores") 