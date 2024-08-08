import os
import math
import torch
import numpy as np
import pickle as pkl

from src.stanza_srilm import NLPTextAnalyzer


from src.utils import nlp, prompt_for_target_inferring, prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring, prompt_for_implicitness_inferring
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random


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
            # --------- Save to Drive
            try:
                colab_path = os.path.join(self.config.preprocess_dir_colab,
                                    '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
                pkl.dump(self.data, open(colab_path, 'wb'))
                print('PREPROCESSED DATA SAVED:', colab_path)
            except:
                print('Failed to save preprocessed data to drive!')
                #--------- Save to Drive


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
            implicitness_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                prompt = prompt_for_implicitness_inferring(line)
                implicitness_tokens.append(prompt)

            # Given the sentence "the system it comes with does not work properly, so when trying to fix the problems with it it started not working at all.",  Detect if implict speech is being used to express an opinion about a target in the sentence Consider - Contextual Dependence: For example, the phrase "Try the tandoori salmon!" lacks explicit sentiment words, but the recommendation implies a positive sentiment based on cultural understanding and context. - Absence of Direct Opinion Expression: For example, "The new mobile phone can just fit in my pocket" implies a positive sentiment about the phones portability without using explicit positive adjectives. - Irony or Sarcasm: For example, saying "What a wonderful day!" in the middle of a storm conveys a negative sentiment through irony. - Dependence on Pragmatic Theories: For instance, a polite statement like "Its not the best service Ive experienced" might imply dissatisfaction, though it appears mild or neutral on the surface. - Multi-Hop Reasoning: For instance, the statement "The book was on the top shelf" might require reasoning about the inconvenience of reaching it to infer a negative sentiment. Return a "True" or "False" boolean if implicit speech is being used regardless of its polarity.
            batch_implicitness_input = self.tokenizer.batch_encode_plus(implicitness_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_implicitness_input = batch_implicitness_input.data

            target_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                prompt = prompt_for_target_inferring(line)
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

            res = {
                'inferred_implicitness_prompt_ids': batch_implicitness_input['input_ids'],  # Full Prompt 4 implicitness Extraction
                'inferred_implicitness_prompt_masks': batch_implicitness_input['attention_mask'],
                'inferred_target_ids': batch_target_input['input_ids'],# Full Prompt 4 Target Extraction
                'inferred_target_masks': batch_target_input['attention_mask'],

                'input_ids': batch_input['input_ids'],# aspect_ids: Full Prompt 4 Aspect of Target Extraction
                'input_masks': batch_input['attention_mask'], # aspect_masks
                'context_A_ids': batch_contexts_A['input_ids'],# encoded prompt context, "i already have a hp laptop i bought last year that's standard size.", '
                'context_A_masks': batch_contexts_A['attention_mask'],
                'target_ids': batch_targets['input_ids'],# Aspect Term Ids ('size', 'programs', 'applications', 'Windows', 'price', 'color reproduction', 'case', 'battery life', 'retail price', 'carry')
                'target_masks': batch_targets['attention_mask'],# Aspect Term Masks

                'output_ids': batch_output['input_ids'],# raw sentiment label
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),# 0, 1, 2
                'implicits': torch.tensor(implicits),# 0,1
                #'upos_ids': upos_ids, #0,1
                #'head_ids': head_ids,  # 0,1
                #'deprel_ids': deprel_ids,  # 0,1
                #'ner_ids': ner_ids,  # 0,1
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'

"""
This preprocessor with Stanza took:
    Laptops: 1270 + 53 + 401 seconds/21 mins
    Restaurants:
    Debug:
"""

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.NLPanalyzer = NLPTextAnalyzer()
        #self.nlp = spacy.load("en_core_web_lg")

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


    def new_transformer2indices(self, cur_data):
        comments = cur_data['raw_texts']
        #nlp_data = []
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
            #ner = nlp_data[i]['ner_list']

            doc = nlp(text)
            ner = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

            #res.append([text, target, label, implicit])
            res.append([text, target, label, implicit, upos, heads, deprel, ner])
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            #data = self.transformer2indices(dataset[i])
            data = self.new_transformer2indices(dataset[i])
            res.append(data)
        return res