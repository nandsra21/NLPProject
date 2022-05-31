from transformers import T5Tokenizer, T5ForConditionalGeneration

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained('./home/nandsra/model')

tokenizer = T5Tokenizer.from_pretrained("t5-small")
special_tokens_dict = {'additional_special_tokens': ['[BOS]', '[CLS]', '[OFFY]', '[OFFN]', '[IND]', '[STE]', '[END]', '[GRP]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

task_prefix = 'translate English to German: '
sentences = ['The house is wonderful.', 'I like to work in NYC.'] # use different length sentences to test batching
inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    do_sample=False, # disable sampling to test if batching affects output
)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))





3:35
# ['Das Haus ist wunderbar.', 'Ich arbeite gerne in NYC.']