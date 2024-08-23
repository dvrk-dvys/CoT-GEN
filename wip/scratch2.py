from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
# training
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
#labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
#outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)

#-----------------------------------
#!!a fake output just to get the current encoding!!
decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=decoder_input_ids, output_hidden_states=True)
encoder_embeddings = model.encoder.embed_tokens(decoder_input_ids)
print(encoder_embeddings)
#-----------------------------------


loss = outputs.loss
logits = outputs.logits
print(loss)
print(logits)


print(outputs.encoder_hidden_states)
print(outputs.decoder_hidden_states)


#assert torch.allclose(encoder_embeddings, outputs.encoder_hidden_states[0], atol=1e-5), "Embeddings do not match!"

#print("Embeddings match!")

