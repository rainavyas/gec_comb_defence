from greco.models import GRECO
import torch
from src.tools.tools import get_default_device
from tqdm import tqdm

class GRECOcombiner:
    def __init__(self, source_sentences, pred_texts, run_comb=True):
        device = get_default_device()
        self.model = GRECO('microsoft/deberta-v3-large')
        self.model.load_state_dict(torch.load('greco/models/checkpoint.bin', map_location=torch.device('cpu')))
        self.model.to(device)
        if run_comb:
            self.combined_texts = self._make_all_changes(source_sentences, pred_texts)

    def _make_all_changes(self, source_sentences, pred_texts):
        selected_samples = []
        # for n, samples in tqdm(enumerate(zip(*pred_texts)), total=len(source_sentences)):
        for n, samples in enumerate(zip(*pred_texts)):
            source = source_sentences[n]
            scores = [self.model.score([source], [sample]) for sample in samples]
            ind = scores.index(max(scores))
            selected_samples.append(samples[ind])
        return selected_samples