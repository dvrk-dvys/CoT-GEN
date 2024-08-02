import os
import math
import torch
import numpy as np
import pickle as pkl


from src.utils import prompt_for_target_inferring, prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random

from src.stanza_srilm import NLPTextAnalyzer
#from pyspark.sql import SparkSession
#from pyspark.shell import sc


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]

        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        try:
            input_tokens, input_targets, input_labels, implicits = zip(*data)
        except:
             print('error: int object not iterable')
        if self.config.reasoning == 'prompt':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot == True:
                    _, prompt = prompt_direct_inferring(line, input_targets[i])
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':
            #--------
            target_tokens = []
            contexts_Z = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step_0, prompt = prompt_for_target_inferring(line)
                contexts_Z.append(context_step_0)
                target_tokens.append(prompt)

            # Given the sentence "the gray color was a good choice.", identify the target (entitiy or subject) being discussed. The target might be explicitely mentioned in the text or referred to indirectly. If the target is not explicitly mentioned select the most appropriate approximation of the Target entity type from this Named Entity Recognition Vocabulary: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
            batch_target_input = self.tokenizer.batch_encode_plus(target_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_target_input = batch_target_input.data
            #--------

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                contexts_A.append(context_step1)
                new_tokens.append(prompt)

            #'Given the sentence "the gray color was a good choice.", '
            batch_contexts_A = self.tokenizer.batch_encode_plus(contexts_A, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_A = batch_contexts_A.data

            #'Given the sentence "the gray color was a good choice.", which specific aspect of BATTERY is possibly mentioned?'
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            #'gray color'
            batch_targets = self.tokenizer.batch_encode_plus(list(input_targets), padding=True, return_tensors='pt',
                                                             max_length=self.config.max_length)
            batch_targets = batch_targets.data

            # 0,1,2
            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data
            print("Input IDs shape:",  batch_input['input_ids'].shape)
            print("Attention Mask shape:", batch_input['attention_mask'].shape)


            res = {
                'inferred_target_ids': batch_target_input['input_ids'],
                'inferred_target_masks': batch_target_input['attention_mask'],
                'aspect_ids': batch_input['input_ids'],
                'aspect_masks': batch_input['attention_mask'],
                'context_A_ids': batch_contexts_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits) #0,1
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        dataname = self.config.dataname
        train_file = os.path.join(self.config.data_dir, dataname,
                                  '{}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_file = os.path.join(self.config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        train_data = pkl.load(open(train_file, 'rb'))
        test_data = pkl.load(open(test_file, 'rb'))
        ids = np.arange(len(train_data))
        np.random.shuffle(ids)
        total_length = len(next(iter(train_data.values())))
        lens = min(150, total_length // 2)  # original lenth: 150
        #lens = 150
        valid_data = {w: v[-lens:] for w, v in train_data.items()}
        train_data = {w: v[:-lens] for w, v in train_data.items()}

        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append([text, target, label, implicit])
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            data = self.transformer2indices(dataset[i])
            res.append(data)
        return res



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
    'PRODUCT': 15, 'QUANTITY': 16, 'TIME': 17, 'WORK_OF_ART': 18
} # 'NONE': 19

class NewDataLoader:
    def __init__(self, config):
        self.config = config
        config.NewPreprocessor = NewPreprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        #if os.path.exists(path):
        #    self.data = pkl.load(open(path, 'rb'))
        #else:
        self.data = self.config.NewPreprocessor.forward()
        pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]

        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        try:
            #input_tokens, input_targets, input_labels, implicits = zip(*data)
            input_tokens, input_targets, input_labels, implicits, upos_ids, head_ids, deprel_ids, ner_ids = zip(*data)

        except Exception as e:
             print(f'Error: {e}')
        if self.config.reasoning == 'prompt':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot == True:
                    _, prompt = prompt_direct_inferring(line, input_targets[i])
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                contexts_A.append(context_step1)
                new_tokens.append(prompt)

            batch_contexts_A = self.tokenizer.batch_encode_plus(contexts_A, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_A = batch_contexts_A.data
            batch_targets = self.tokenizer.batch_encode_plus(list(input_targets), padding=True, return_tensors='pt',
                                                             max_length=self.config.max_length)
            batch_targets = batch_targets.data
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'context_A_ids': batch_contexts_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'


class NewPreprocessor:
    def __init__(self, config):
        self.config = config
        self.NLPanalyzer = NLPTextAnalyzer()
        #self.spark_session = SparkSession.builder.master("local[*]").appName("NLP_Loader").getOrCreate()


    def read_file(self):
        dataname = self.config.dataname
        train_file = os.path.join(self.config.data_dir, dataname,
                                  '{}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_file = os.path.join(self.config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))

        #try:
        train_data = pkl.load(open(train_file, 'rb'))
        test_data = pkl.load(open(test_file, 'rb'))
        #except:
            #train_data = sc.pickleFile(train_file).collect()
            #train_data = self.spark_session.createDataFrame(train_pickleRdd)
            #test_data = sc.pickleFile(train_file).collect()
            #test_data = self.spark_session.createDataFrame(test_pickleRdd)


        ids = np.arange(len(train_data))
        np.random.shuffle(ids)
        total_length = len(next(iter(train_data.values())))
        lens = min(150, total_length // 2) #original lenth: 150
        valid_data = {w: v[-lens:] for w, v in train_data.items()}
        train_data = {w: v[:-lens] for w, v in train_data.items()}

        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        comments = cur_data['raw_texts']
        nlp_data = []
        nlp_data = self.NLPanalyzer.nlp_processor(comments)

        res = []
        for i in range(len(comments)):
            text = comments[i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)

            upos = nlp_data[i]['upos_list']
            heads = nlp_data[i]['head_list']
            deprel = nlp_data[i]['deprel_list']
            ner = nlp_data[i]['ner_list']

            #res.append([text, target, label, implicit])
            res.append([text, target, label, implicit, upos, heads, deprel, ner])
        return res
    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()

        res = []
        for i, mode in enumerate(modes):
            data = self.transformer2indices(dataset[i])
            res.append(data)
        return res