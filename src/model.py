import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration


class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]

        #output_ids = output_ids.to("cpu")
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        #output_ids = output_ids.to(self.config.device)
        #input_ids = input_ids.to(self.config.device)
        #input_masks = input_masks.to(self.config.device)
        #output_masks = output_masks.to(self.config.device)

        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output

    def head_embeddings(self, input_ids, output_ids):
        prompts = [self.tokenizer.decode(ids) for ids in input_ids]
        prompts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in prompts]

        targets = [self.tokenizer.decode(ids) for ids in output_ids]
        targets = [word.replace('<pad>', '').replace('</s>', '').strip() for word in targets]


        for i, _ in enumerate(prompts):
            if (targets[i] not in _) and (targets[i] != 'None'):
                # Concatenate the target word to the input sentence if its not present in the sentence
                disembodied_head = _ + targets[i]  # + ' target: ' * target_word
                prompts[i] = disembodied_head
                #disembodied_head = self.tokenizer.decode(input_ids) + ' ' + targets[i]  # + ' target: ' * target_word
                #input_ids = self.tokenizer(disembodied_head, return_tensors="pt").input_ids
                #input_masks = self.tokenizer(disembodied_head, return_tensors="pt").attention_mask

        input_ids = self.tokenizer(prompts, padding=True, return_tensors="pt", max_length=self.config.max_length).input_ids.to(self.config.device)

        # !!a fake output just to get the current encoding!!
        batch_size = input_ids.size(0)
        decoder_input_ids = self.tokenizer("<pad>", padding=True, return_tensors="pt", max_length=self.config.max_length).input_ids.to(self.config.device)
        decoder_input_ids = decoder_input_ids.repeat(batch_size, 1)
        outputs = self.engine(input_ids=input_ids, labels=decoder_input_ids, output_hidden_states=True)

        # Pass through the encoder to get the hidden states (embeddings)
        hidden_states = outputs.encoder_hidden_states[-1]

        batch_target_embeddings = []
        for i in range(len(input_ids)):
            tokens = [self.tokenizer.decode(ids) for ids in output_ids]
            tokens = [t.replace('<pad>', '').replace('</s>', '').strip() for t in tokens]

            #tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            target_index = tokens[0].index(targets[0])

            # Extract the embedding corresponding to the target word
            target_embedding = hidden_states[:, target_index, :]
            batch_target_embeddings.append(target_embedding)
        batch_target_embeddings = torch.stack(batch_target_embeddings, dim=0)

        return batch_target_embeddings


    #def single_head_embeddings(self, input_ids, output_ids):
    #    prompts = [self.tokenizer.decode(ids) for ids in input_ids]
    #    prompts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in prompts]
#
    #    targets = [self.tokenizer.decode(ids) for ids in output_ids]
    #    targets = [word.replace('<pad>', '').replace('</s>', '').strip() for word in targets]

    #    test = 'Given the sentence "also it comes with very useful applications like iphoto that it is the best photo application i have ever had",'
    #    correct_input_ids = self.tokenizer(prompts[0], padding=True, return_tensors="pt", max_length=self.config.max_length).input_ids.to(self.config.device)

        # !!a fake output just to get the current encoding!!
    #    decoder_input_ids = self.tokenizer("<pad>", padding=True, return_tensors="pt", max_length=self.config.max_length).input_ids.to(self.config.device)

    #    outputs = self.engine(input_ids=correct_input_ids, labels=decoder_input_ids, output_hidden_states=True)
    #    encoder_embeddings = self.engine.encoder.embed_tokens(decoder_input_ids)
    #    print(encoder_embeddings)

    #    loss = outputs.loss
    #    logits = outputs.logits
    #    print(loss)
    #    print(logits)

    #    print(outputs.encoder_hidden_states)
    #    print(outputs.decoder_hidden_states)

        # Pass through the encoder to get the hidden states (embeddings)
    #    hidden_states = outputs.encoder_hidden_states[0]

    #    tokens = [self.tokenizer.decode(ids) for ids in output_ids]
    #    tokens = [t.replace('<pad>', '').replace('</s>', '').strip() for t in tokens]

        #tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
    #    target_index = tokens[0].index(targets[0])

        # Extract the embedding corresponding to the target word
    #    target_embedding = hidden_states[:, target_index, :]

    #    return target_embedding
