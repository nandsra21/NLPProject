from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from IPython import embed

output_dir = '/home/nandsra'
model_dir = output_dir + '/model'
model = T5ForConditionalGeneration.from_pretrained('./home/nandsra/model_regular')#_scrambled')

#tokenizer = T5Tokenizer.from_pretrained("t5-small")
#special_tokens_dict = {'additional_special_tokens': ['[BOS]', '[CLS]', '[OffY]', '[OffN]', '[IND]', '[STE]', '[END]', '[GRP]']}
#num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#model.resize_token_embeddings(len(tokenizer))
tokenizer = T5Tokenizer.from_pretrained('./home/nandsra/model_regular/tokenizer')

labels = ['[BOS] white revolution is the only solution [CLS]',
 '[BOS] Kendrick just dropped the coldest video of all fucking time [CLS]',
 '[BOS] What a fucking stupid bitch shut the fuck up already [CLS] ',
 '[BOS] need to fuck sum bad , I got some pressure built up :P [CLS] ',
 '[BOS] I take a bitch home give her long dick [CLS] ',
 '[BOS] You annoyed me every day for months you’re a fucking moron [CLS] ',
 '[BOS] My problem with Korean artists: I dont know how to pronounce your name I can’t hanggul to save my life [CLS] ',
 '[BOS] I’ll continue this thread till non-lifting fags stop lifting like idiots at the gym... Which is never. [CLS] ',
 '[BOS] I thought drugs were the only things black people could shoot up Boy was I wrong [CLS] ',
 '[BOS] yes you are fine in a white majority country. how is mexico doing these days? people come here because you couldn’t build it. [CLS] ',
 '[BOS] black people only seem to thrive in white countries. how strange [CLS] ',
 '[BOS] soon white people will not be a majority anywhere. you believe [CLS] ',
 '[BOS] muslims are always wanting to kill someone! its their duty to kill the infidel. [CLS] ',
 '[BOS] oh how i wish a man with a gun was there! wake up white people! they hate us [CLS] ']

labels = pd.read_csv('regular.dev.csv')["input"].to_list()[:2]
#sentences = ['The house is wonderful.', 'I like to work in NYC.'] # use different length sentences to test batching
inputs = tokenizer(labels, add_special_tokens=False, return_tensors="pt", padding=True)
print(inputs["input_ids"])

output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    do_sample=False, # disable sampling to test if batching affects output
    eos_token_id = tokenizer.convert_tokens_to_ids("[eoo]"),
    max_length = 512
)
embed()
exit()
#print(output_sequences)
#print(tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False))

output = pd.DataFrame({'Source Text': labels, 'Generated Text': tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False)})
output.to_csv(output_dir + "/predictions_regular.csv")
output.to_csv("predictions_regular.csv")

print("just saved predictions")





# write script to calculate accuracy
# SK learn classification metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# f1 score, accuracy score (offensive), only accuracy score for groups
# calculate accuracy (of the offensive label, of the group label <- ask about it) ignore OffN for GRP for SPF dev set
# did it pick the correct target group (binary)
# get familiar for dynahate (annotate from there) / use their model
# crowdsource responses about hate speech

# performance of the model on dynaHate
# READ paper maarten sent on implicit hate (michael)
# SPF dev, implicit hate, dynahate (report on which group is being targeted)
# demo of the model (?) -> create demo using Maartens repo

# merge latent hate and social bias frames

# given a post, get a robust answer


# input_ids = tokenizer([], return_tensors='pt').input_ids
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# # Das Haus ist wunderbar.