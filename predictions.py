from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from IPython import embed

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

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=False, # disable sampling to test if batching affects output
        eos_token_id = tokenizer.convert_tokens_to_ids("[eoo]"),
        max_length = 512
    )
    embed()
    exit()

    output = pd.DataFrame({'Source Text': labels, 'Generated Text': tokenizer.batch_decode(output_sequences, skip_special_tokens=False, padding=False)})
    output.to_csv(args.output)

    print("just saved predictions")




