from transformers import GPT2Tokenizer, GPT2Model
import pandas as pd
from datasets import load_dataset
from IPython import embed
import random
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", nargs = "+", help="the model file(s)")
    parser.add_argument("--input-validation", nargs="+", help="comma separated file with an input and output row")
    parser.add_argument("--output", nargs="+", help="the file of predictions")

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {
        'additional_special_tokens': ['[boi]', '[eoi]', '[OffY]', '[OffN]', '[ind]', '[grp]', '[ste]', '[boo]',
                                      '[eoo]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    output_df = args.output

    i = 0
    for model_file in args.input_model:
        for df in args.input_validation:
            model = GPT2Model.from_pretrained("gpt2")
            model.load_state_dict(torch.load(model_file))
            model.resize_token_embeddings(len(tokenizer))
            model.eval()

            print(model_file + " model file\n")
            labels = pd.read_csv(df)["input"].to_list()
            inputs = tokenizer(labels, add_special_tokens=False, return_tensors="pt", padding=True).to(device)
            print(df + " validation csv\n")
            print("generating predictions...\n")


            output_sequences = model.to(device).generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False, # disable sampling to test if batching affects output
                #eos_token_id = tokenizer.convert_tokens_to_ids("[eoo]"),
                max_length = 64,
                #repetition_penalty = 0.75,
                # setting ngram repetition penalty to 3 (to be congruent with our similarity score acc. values)
                #no_repeat_ngram_size = 5,
                # Adding beam search instead of greedy decoding
                num_beams = 5,
                early_stopping = True
            )
            print(tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False))
            output = pd.DataFrame({'Source Text': labels, 'Generated Text': tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False)})
            output.to_csv(output_df[i])
            print(output_df[i] + " outputdf\n")
            print("just saved predictions")
            i = i + 1




