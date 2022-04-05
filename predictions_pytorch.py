from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from datasets import load_dataset
from IPython import embed
import random
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-model", help="the model file")
    parser.add_argument("--input-model", nargs = "+", help="the model file(s)")
    parser.add_argument("--input-validation", nargs="+", help="comma separated file with an input and output row")
    parser.add_argument("--output", nargs="+", help="the file of predictions")

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.parent_model + "/tokenizer")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    output_df = args.output

    i = 0
    for model_file in args.input_model:
        for df in args.input_validation:

            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            model.load_state_dict(torch.load(model_file))
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
                repetition_penalty = 0.75,
                # setting ngram repetition penalty to 3 (to be congruent with our similarity score acc. values)
                no_repeat_ngram_size = 3,
                # Adding beam search instead of greedy decoding
                #num_beams = 5,
                #early_stopping = True
            )

            output = pd.DataFrame({'Source Text': labels, 'Generated Text': tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False)})
            output.to_csv(output_df[i])
            print(output_df[i] + " outputdf\n")
            print("just saved predictions")
            i = i + 1




