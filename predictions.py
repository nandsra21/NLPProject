from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from datasets import load_dataset
from IPython import embed
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", help="the model file")
    parser.add_argument("--input-validation", help="comma separated file with an input and output row")
    parser.add_argument("--output", help="the file of predictions")

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.input_model + "/tokenizer")
    model = T5ForConditionalGeneration.from_pretrained(args.input_model)

    labels = pd.read_csv(args.input_validation)["input"].to_list()
    inputs = tokenizer(labels, add_special_tokens=False, return_tensors="pt", padding=True)
    print(inputs["input_ids"])

    def build_batches(sentences, batch_size):
        batch_ordered_sentences = list()
        while len(sentences) > 0:
            to_take = min(batch_size, len(sentences))
            select = random.randint(0, len(sentences) - to_take)
            batch_ordered_sentences += sentences[select:select + to_take]
            del sentences[select:select + to_take]
        return batch_ordered_sentences

    def batch_iterator(batch_size=32):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]

    #dataset = pd.read_csv(args.input_validation, on_bad_lines='skip')["input"].to_list()
    #embed()
    #inputs = tokenizer.train_from_iterator(batch_iterator(), add_special_tokens=False, return_tensors="pt", padding=True)

    #dataset = pd.read_csv(args.input_validation)["input"].to_list();
    #batched = build_batches(dataset, batch_size=32)
    #inputs = tokenizer(batched, add_special_tokens=False, return_tensors="pt", padding=True)
    #inputs = batched.apply(lambda examples: tokenizer(examples['input'], add_special_tokens=False, return_tensors="pt", padding=True)
    #                    , batched=True, batch_size=32)


    print("generating predictions...\n")
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=False, # disable sampling to test if batching affects output
        #eos_token_id = tokenizer.convert_tokens_to_ids("[eoo]"),
        max_length = 64,
        repetition_penalty = 0.75
        # setting ngram repetition penalty to 3 (to be congruent with our similarity score acc. values)
        no_repeat_ngram_size = 3,
        # Adding beam search instead of greedy decoding
        num_beams = 5,
        early_stopping = True
    )

    output = pd.DataFrame({'Source Text': labels, 'Generated Text': tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False)})
    output.to_csv(args.output)

    print("just saved predictions")




