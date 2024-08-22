import os
import sys
import torch
import numpy as np
import torch.nn as nn
import logging

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from src.utils import nlp, ner_vocab, prompt_for_opinion_inferring, prompt_for_polarity_inferring, prompt_for_polarity_label
from IPython.display import display, clear_output

#from torch.cuda.amp import autocast, GradScaler
#scaler = GradScaler()
try:
    import google.colab
    in_colab = True
except ImportError:
    in_colab = False




class PromptTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader, start_epoch=0, best_score=0) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''
        self.start_epoch = start_epoch
        self.best_score = best_score

        self.scores, self.lines = [], []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.start_epoch, self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.re_init()
            score = result['default']

            self.add_instance(result)

            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)

                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.engine.cpu().state_dict(), 'best_score': best_score},
                           save_name)
                print(save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        res = self.final_evaluate(best_iter)
        score = res['default']
        self.add_instance(res)

        save_name = self.save_name.format(epoch)

        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        losses = []
        for i, data in enumerate(train_data):
            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                self.add_output(data, output)
        result = self.report_score(mode=mode)
        return result

    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        loaded_state_dict = self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits']

    def add_output(self, data, output):
        is_implicit = data['implicits'].tolist()
        gold = data['input_labels']
        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()
            else:
                if i == 1:
                    ids = np.argwhere(np.array(is_implicit) == 0).flatten()
                else:
                    ids = np.argwhere(np.array(is_implicit) == 1).flatten()
                self.preds[key] += [output[w] for w in ids]
                self.golds[key] += [gold.tolist()[w] for w in ids]

    def report_score(self, mode='valid'):
        res = {}
        res['Acc_SA'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1_SA'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1, 2], average='macro')
        res['F1_ESA'] = f1_score(self.golds['explicits'], self.preds['explicits'], labels=[0, 1, 2], average='macro')
        res['F1_ISA'] = f1_score(self.golds['implicits'], self.preds['implicits'], labels=[0, 1, 2], average='macro')
        res['default'] = res['F1_SA']
        res['mode'] = mode
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res


class ThorTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader, start_epoch=0, best_score=0) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.save_name_colab = os.path.join(config.target_dir_colab, config.save_name)
        self.final_score = 0
        self.final_res = ''
        self.start_epoch = start_epoch
        self.best_score = best_score

        self.scores, self.lines = [], []
        self.re_init()
        #self.nlp = spacy.load("en_core_web_lg")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()


    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.start_epoch, self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.re_init()
            score = result['default']
            self.add_instance(result)
            _res = self.get_best()

            message = f'EPOCH {epoch} Eval:', result
            print(message, flush=True)
            self.logger.info(message)

            self.add_instance(result)
            _res = self.get_best()

            #if epoch == 2:
            #    print()
            #elif epoch == 1:
            #    print()
            #elif epoch == 0:
            #    print()

            if in_colab:
                print('Colab environment', flush=True)
            else:
                print('Local environment', flush=True)



            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           save_name)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f'MODEL SAVED at {current_time}: {save_name}'
                print(message, flush=True)
                self.logger.info(message)
                self.model.to(self.config.device)
                #--------- Save to Drive
                if in_colab:
                    save_name_colab = self.save_name_colab.format(epoch)
                    if not os.path.exists(self.config.target_dir_colab):
                        os.makedirs(self.config.target_dir_colab)
                    torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                               save_name_colab)

                    print('MODEL SAVED to Drive:', save_name_colab, flush=True)
                    self.model.to(self.config.device)
                #--------- Save to Drive


            elif epoch - best_iter > self.config.patience:
                # print("Not upgrade for {} steps, early stopping...".format(self.config.patience), flush=True)
                message = f"Not upgrade for {self.config.patience} steps, early stopping..."
                print(message, flush=True)
                self.logger.info(message)
                break
            self.model.to(self.config.device)

        res = self.final_evaluate(best_iter)
        score = res['default']
        self.add_instance(res)
        save_name = self.save_name.format(epoch)
        self.final_score, self.final_res = score, res

    #--------------------------------------------------------------------------------------------------------

    def calc_vector_dist(self, target, approximation):
        #word_vec = self.nlp(target).vector
        #similarities = {}
        #for approx in approximations:
        #similarities[approx] = self.nlp(approx).similarity(self.nlp(target))
        #max(similarities, key=similarities.get)
        test = nlp(approximation).similarity(nlp(target))
        return test

    def calc_approximate_vector_weights(self, approximations, targets):
        # approximate_targets = ['ONE', 'LANGUAGE', 'PRODUCT', 'None', 'NONE', 'EVENT', 'None', 'None', 'NONE', 'NONE']
        # targets = ['system', 'gray color', 'webcam', 'Windows XP SP2', 'service', 'Games', 'gaming', 'support', 'software', 'screen']
        loss_weights = []
        for i, t in enumerate(approximations):
            if t.upper() != 'NONE':
                vector_similarity = self.calc_vector_dist(targets[i], t)
                loss_weights.append(vector_similarity)
            else:
                loss_weights.append(0.0)
        return loss_weights

    def prepare_step_zero(self, **kwargs):
        inferred_target_prompt_ids, inferred_target_prompt_masks, target_ids, target_masks, context_A_ids, context_A_masks, inferred_implicitness_prompt_ids, inferred_implicitness_prompt_masks, implicits = [kwargs[w] for w in
        'inferred_target_ids, inferred_target_masks, target_ids, target_masks, context_A_ids, context_A_masks, inferred_implicitness_prompt_ids, inferred_implicitness_prompt_masks, implicits'.strip().split(', ')]

        # Infer Target
        prompts = [self.model.tokenizer.decode(ids) for ids in inferred_target_prompt_ids]
        prompts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in prompts]
        #print(prompts[0])# Given the sentence "my opinion of sony has been dropping as fast as the stock market, given their horrible support, but this machine just caused another plunge.", Your task is to identify the **target** being discussed in the sentence. The target could be explicitly mentioned (e.g., a product, service, feature, person, topic, idea, etc.) or it might be implied through context (implicit). In cases where the target is implicit, infer the most likely entity type based on the context provided. Consider any descriptory words, aspect terms or opinion expressions that may be depending on and pointing to the target. Use this Named Entity Recognition Vocabulary: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART

        labeled_targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        labeled_targets = [target.replace('<pad>', '').replace('</s>', '').strip() for target in labeled_targets]
        #print(labeled_targets[0])# 'support'

        target_res = {
            'input_ids': inferred_target_prompt_ids, # Given the sentence "the system it comes with does not work properly, so when trying to fix the problems with it it started not working at all.",  Detect if implict speech is being used to express an opinion about a target in the sentence Consider - Contextual Dependence: For example, the phrase "Try the tandoori salmon!" lacks explicit sentiment words, but the recommendation implies a positive sentiment based on cultural understanding and context. - Absence of Direct Opinion Expression: For example, "The new mobile phone can just fit in my pocket" implies a positive sentiment about the phones portability without using explicit positive adjectives. - Irony or Sarcasm: For example, saying "What a wonderful day!" in the middle of a storm conveys a negative sentiment through irony. - Dependence on Pragmatic Theories: For instance, a polite statement like "Its not the best service Ive experienced" might imply dissatisfaction, though it appears mild or neutral on the surface. - Multi-Hop Reasoning: For instance, the statement "The book was on the top shelf" might require reasoning about the inconvenience of reaching it to infer a negative sentiment. Return a "True" or "False" boolean if implicit speech is being used regardless of its polarity.
            'input_masks': inferred_target_prompt_masks,
            'output_ids': target_ids, # ['system', 'gray color', 'webcam', 'Windows XP SP2', 'service', 'Games', 'gaming', 'support', 'software', 'screen']
            'output_masks': target_masks,
        }

        implicits_list = implicits.tolist()
        implicit_labels = list(map(lambda i: "True" if i == 1 else "False", implicits_list))
        #print(implicit_labels[0])

        inferred_targets = self.model.generate(**target_res)# ['WORK_OF_ART', 'WORK_OF_ART', 'WORK_OF_ART', 'PC', 'WORK_OF_ART', 'games', 'LANGUAGE', 'Sony', 'Software', 'LANGUAGE']
        #approx_vector_weights = self.calc_approximate_vector_weights(inferred_targets, labeled_targets)

        inferred_targets = self.model.tokenizer.batch_encode_plus(inferred_targets, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        inferred_targets = inferred_targets.data['input_ids']
        approx_embeddings = self.model.head_embeddings(context_A_ids, inferred_targets)
        target_embeddings = self.model.head_embeddings(context_A_ids, target_ids)

        batch_implicit_labels = self.model.tokenizer.batch_encode_plus(implicit_labels, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_implicit_labels = batch_implicit_labels.data

        implicitness_res = {
            'input_ids': inferred_implicitness_prompt_ids,# Given the sentence "the system it comes with does not work properly, so when trying to fix the problems with it it started not working at all.",  Detect if implict speech is being used to express an opinion about a target in the sentence Consider - Contextual Dependence: For example, the phrase "Try the tandoori salmon!" lacks explicit sentiment words, but the recommendation implies a positive sentiment based on cultural understanding and context. - Absence of Direct Opinion Expression: For example, "The new mobile phone can just fit in my pocket" implies a positive sentiment about the phones portability without using explicit positive adjectives. - Irony or Sarcasm: For example, saying "What a wonderful day!" in the middle of a storm conveys a negative sentiment through irony. - Dependence on Pragmatic Theories: For instance, a polite statement like "Its not the best service Ive experienced" might imply dissatisfaction, though it appears mild or neutral on the surface. - Multi-Hop Reasoning: For instance, the statement "The book was on the top shelf" might require reasoning about the inconvenience of reaching it to infer a negative sentiment. Return a "True" or "False" boolean if implicit speech is being used regardless of its polarity.
            'input_masks': inferred_implicitness_prompt_masks,
            'output_ids': batch_implicit_labels['input_ids'],# ['False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'True', 'False']
            'output_masks': batch_implicit_labels['attention_mask'],
        }

        target_res = {k: v.to(self.config.device) for k, v in target_res.items()}
        implicitness_res = {k: v.to(self.config.device) for k, v in implicitness_res.items()}
        return target_res, implicitness_res, approx_embeddings, target_embeddings #approx_vector_weights

    def prepare_step_one(self, **kwargs):
        #'aspect_ids': batch_input['input_ids'],
        #'aspect_masks': batch_input['attention_mask'],
        #'context_A_ids': batch_contexts_A['input_ids'],

        aspect_ids, aspect_masks, context_A_ids = [kwargs[w] for w in 'input_ids, input_masks, context_A_ids'.strip().split(', ')]
        #aspect
        targets = [self.model.tokenizer.decode(ids) for ids in aspect_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]
        #print(targets[0])
        contexts_A = [self.model.tokenizer.decode(ids) for ids in context_A_ids]
        contexts_A = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_A]
        #print(contexts_A[0])

        res = {
            'input_ids': aspect_ids,
            'input_masks': aspect_masks,
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    # --------------------------------------------------------------------------------------------------------

    def prepare_step_two(self, aspect_exprs, data):
        context_A_ids, target_ids = [data[w] for w in 'context_A_ids, target_ids'.strip().split(', ')]
        #aspect
        contexts_A = [self.model.tokenizer.decode(ids) for ids in context_A_ids]
        contexts_A = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_A]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_B = []
        for context, target, aspect_expr in zip(contexts_A, targets, aspect_exprs):
            context_B, prompt = prompt_for_opinion_inferring(context, target, aspect_expr)
            new_prompts.append(prompt)
            contexts_B.append(context_B)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data
        batch_contexts_B = self.model.tokenizer.batch_encode_plus(contexts_B, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_B = batch_contexts_B.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_B_ids': batch_contexts_B['input_ids'],
            'target_ids': target_ids,
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_three(self, opinion_exprs, data):
        context_B_ids, target_ids = [data[w] for w in 'context_B_ids, target_ids'.strip().split(', ')]
        contexts_B = [self.model.tokenizer.decode(ids) for ids in context_B_ids]
        contexts_B = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_B]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_C = []
        for context, target, opinion_expr in zip(contexts_B, targets, opinion_exprs):
            context_C, prompt = prompt_for_polarity_inferring(context, target, opinion_expr)
            new_prompts.append(prompt)
            contexts_C.append(context_C)

        batch_contexts_C = self.model.tokenizer.batch_encode_plus(contexts_C, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_C = batch_contexts_C.data
        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_C_ids': batch_contexts_C['input_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_sentiment_label(self, polarity_exprs, pre_cxt, data):
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]
        #output_ids are the final overall polarity of the input text from the db

        context_C_ids = pre_cxt['context_C_ids']
        contexts_C = [self.model.tokenizer.decode(ids) for ids in context_C_ids]
        contexts_C = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_C]

        new_prompts = []
        for context_C, polarity_expr in zip(contexts_C, polarity_exprs):
            prompt = prompt_for_polarity_label(context_C, polarity_expr)
            new_prompts.append(prompt)

        #new_prompts = Given the sentence "the gray color was a good choice.", The mentioned aspect is about color. The opinion towards the mentioned aspect of BATTERY is The gray color was a good choice. The sentiment polarity is positive. Based on these contexts, summarize and return the sentiment polarity only, such as positive, neutral, or negative.
        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=3)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length)

        losses = []
        for i, data in enumerate(train_data):
            try:
                #****--------
                target_label_data, implicitness_label_data, approx_embeddings, target_embeddings = self.prepare_step_zero(**data)
                #****--------
                step_one_inferred_data = self.prepare_step_one(**data)
                step_one_inferred_output = self.model.generate(**step_one_inferred_data)

                # Inferred aspect of 'target': color
                #'target':Battery --the target comes from labelled data
                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)

                # Inferred implicit opinion expression of the aspect of the 'target' (Battery): 'The gray color was a good choice'
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                #Context: 'Given the sentence "the gray color was a good choice.", The mentioned aspect is about color.'
                #target: Battery # Opinion Expression: the gray color was a good choice
                step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)

                #'The sentiment polarity is positive'
                step_three_inferred_output = self.model.generate(**step_two_inferred_data)

                #'Given the sentence "the gray color was a good choice.", The mentioned aspect is about color. The opinion towards the mentioned aspect of BATTERY is The gray color was a good choice. The sentiment polarity is positive. Based on these contexts, summarize and return the sentiment polarity only, such as positive, neutral, or negative.'
                step_label_data = self.prepare_sentiment_label(step_three_inferred_output, step_two_inferred_data, data)

                #with autocast():

                #****--------
                target_loss = self.model(**target_label_data)
                #approx_vector_tensor = torch.tensor(approx_vector_weights).to(target_loss.device)
                #weights = 1 - approx_vector_tensor
                #weighted_loss = target_loss * weights.mean()

                cosine_similarity = nn.CosineSimilarity(dim=1)
                similarity_scores = cosine_similarity(approx_embeddings, target_embeddings)
                approximation_loss = 1 - similarity_scores.mean()
                implicitness_loss = self.model(**implicitness_label_data)
                #****--------

                loss = self.model(**step_label_data)

                #****--------
                combined_loss = (
                        self.config.target_loss_alpha * target_loss +
                        self.config.approximation_loss_alpha * approximation_loss +
                        self.config.implicitness_loss_alpha * implicitness_loss +
                        self.config.sentiment_loss_alpha * loss
                )

                losses.append(combined_loss.item())
                #combined_loss.backward()
                #scaler.step(self.config.optimizer)
                #scaler.update()
                #****--------

                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
                train_data.set_description(description)

                self.config.optimizer.step()
                self.config.scheduler.step()
                self.model.zero_grad()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Out of memory error caught. Switching to CPU.")
                    self.config.device = torch.device("cpu")
                    self.model.to(self.config.device)
                    for data_key in data.keys():
                        if isinstance(data[data_key], torch.Tensor):
                            data[data_key] = data[data_key].to(self.config.device)
                else:
                    raise e


    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                target_label_data, implicitness_label_data, approx_embeddings, target_embeddings = self.prepare_step_zero(**data)

                step_one_inferred_data = self.prepare_step_one(**data)
                step_one_inferred_output = self.model.generate(**step_one_inferred_data)

                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
                step_three_inferred_output = self.model.generate(**step_two_inferred_data)

                step_label_data = self.prepare_sentiment_label(step_three_inferred_output, step_two_inferred_data, data)
                output = self.model.evaluate(**step_label_data)# output: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
                self.add_output(data, output, approx_embeddings, target_embeddings)

        result = self.report_score(mode=mode)
        message = "output test: Finish evaluate Step"
        self.logger.info(message)
        print(message, flush=True)
        return result

    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        print(PATH)
        state_dict = torch.load(PATH, map_location=self.config.device)['model']
        #new_state_dict = {'engine.' + k: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits', 'approx'] #'target'

    def add_output(self, data, output, approx_embeddings, target_embeddings):
        is_implicit = data['implicits'].tolist()
        gold = data['input_labels']
        cosine_similarity = nn.CosineSimilarity(dim=1)
        similarity_scores = cosine_similarity(approx_embeddings, target_embeddings)

        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()
            elif i == 3:
                self.preds[key] += similarity_scores.cpu().tolist()
            else:
                if i == 1:
                    ids = np.argwhere(np.array(is_implicit) == 0).flatten()
                else:
                    ids = np.argwhere(np.array(is_implicit) == 1).flatten()
                self.preds[key] += [output[w] for w in ids]
                self.golds[key] += [gold.tolist()[w] for w in ids]


    def report_score(self, mode='valid'):
        res = {}
        res['Acc_SA'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1_SA'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1, 2], average='macro')
        res['F1_ESA'] = f1_score(self.golds['explicits'], self.preds['explicits'], labels=[0, 1, 2], average='macro')
        res['F1_ISA'] = f1_score(self.golds['implicits'], self.preds['implicits'], labels=[0, 1, 2], average='macro')
        #res['F1_TARGET'] = f1_score(self.golds['targets'], self.preds['targets'], labels=[0, 1, 2], average='macro')
        res['Avg_Cosine_Similarity'] = np.mean(self.preds['approx'])  # 'approx' key holds similarity scores

        #res['composite_score'] = (
        #        0.3 * res['F1_SA'] +
        #        0.3 * res['F1_ESA'] +
        #        0.3 * res['F1_ISA'] +
        #        0.1 * res['Avg_Cosine_Similarity']
        #)

        res['composite_score'] = (
                0.7 * res['F1_SA'] + #Overall Sentiment
                0.05 * res['F1_ESA'] + #Label Explicit Sentiment as Explicit
                0.05 * res['F1_ISA'] + #Label Implicit Sentiment as Implicit
                0.1 * res['Avg_Cosine_Similarity']
        )

        #res['default'] = res['F1_SA']
        res['default'] = res['composite_score']
        res['mode'] = mode
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res