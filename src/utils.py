import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from functools import wraps
import time


upos_vocab = {
    'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6,
    'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11,
    'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17
}

deprel_vocab = {
    'acl': 1, 'acl:relcl': 2, 'advcl': 3, 'advcl:relcl': 4, 'advmod': 5,
    'advmod:emph': 6, 'advmod:lmod': 7, 'amod': 8, 'appos': 9, 'aux': 10,
    'aux:pass': 11, 'case': 12, 'cc': 13, 'cc:preconj': 14, 'ccomp': 15,
    'clf': 16, 'compound': 17, 'compound:lvc': 18, 'compound:prt': 19,
    'compound:redup': 20, 'compound:svc': 21, 'conj': 22, 'cop': 23,
    'csubj': 24, 'csubj:outer': 25, 'csubj:pass': 26, 'dep': 27, 'det': 28,
    'det:numgov': 29, 'det:nummod': 30, 'det:poss': 31, 'discourse': 32,
    'dislocated': 33, 'expl': 34, 'expl:impers': 35, 'expl:pass': 36,
    'expl:pv': 37, 'fixed': 38, 'flat': 39, 'flat:foreign': 40, 'flat:name': 41,
    'goeswith': 42, 'iobj': 43, 'list': 44, 'mark': 45, 'nmod': 46,
    'nmod:poss': 47, 'nmod:tmod': 48, 'nsubj': 49, 'nsubj:outer': 50,
    'nsubj:pass': 51, 'nummod': 52, 'nummod:gov': 53, 'obj': 54, 'obl': 55,
    'obl:agent': 56, 'obl:arg': 57, 'obl:lmod': 58, 'obl:tmod': 59, 'orphan': 60,
    'parataxis': 61, 'punct': 62, 'reparandum': 63, 'root': 64, 'vocative': 65,
    'xcomp': 66
}

ner_vocab = {
    'CARDINAL': 1, 'DATE': 2, 'EVENT': 3, 'FAC': 4, 'GPE': 5,
    'LANGUAGE': 6, 'LAW': 7, 'LOC': 8, 'MONEY': 9, 'NORP': 10,
    'ORDINAL': 11, 'ORG': 12, 'PERCENT': 13, 'PERSON': 14,
    'PRODUCT': 15, 'QUANTITY': 16, 'TIME': 17, 'WORK_OF_ART': 18,
    'NONE': 19
}

def prompt_for_target_inferring(context):
    new_context = f'Given the sentence "{context}", '
    instructions = (
        'Your task is to identify the **target** being discussed in the sentence. '
        'The target could be explicitly mentioned (e.g., a product, service, feature, person, topic, idea, etc.) '
        'or it might be implied through context (implicit). '
        'In cases where the target is implicit, infer the most likely entity type based on the context provided if it were explicitly mentioned.'
        'If more than one is likely, pick the top two that fit best.'
        'Consider any descriptory words, aspect terms or opinion expressions that may be depending on and pointing to the target.'
        'If the text contains neither an explicit or possible implicit target an/or named entity, choose "None".'
    )
    ner_vocabulary = ', '.join(list(ner_vocab.keys()))
    prompt = new_context + instructions + f' Use this Named Entity Recognition Vocabulary: {ner_vocabulary}'
    #prompt = new_context + f'Your goal is to identify the target in the sentence that being discussed. The target might be explicitely mentioned as a word or phrase in the text or referred to indirectly through implicit speech pointing toward to sentactically unspecified topic.' \
    #                       f' If the target is not explicitly mentioned in the text select the most appropriate approximation of the Implicit Target entity type using context clues from this Named Entity Recognition Vocabulary:' \
    #                       f' CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART'
    return new_context, prompt


def prompt_direct_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_direct_inferring_masked(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'the sentiment polarity towards {target} is [mask]'
    return new_context, prompt


def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
    return new_context, prompt


def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' The mentioned aspect is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr):
    prompt = context + f' The sentiment polarity is {polarity_expr}.' + ' Based on these contexts, summarize and return the sentiment polarity only, such as positive, neutral, or negative.'
    return prompt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res

def runtime(func):
    @wraps(func)
    def runtime_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return runtime_wrapper