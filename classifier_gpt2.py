from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from tqdm.auto import tqdm
from datasets import load_metric
import re
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
from IPython import embed
# dont include [boo] [eoo] tokens

def BLEUScores(merged_df_preds):
    # get data within the implication bounds
    list_preds = merged_df_preds["generated"].tolist()
    real_val_list = merged_df_preds["output"].tolist()
    import re
    total_val = []
    for i in range(0, len(list_preds)):
        list_preds[i] = list_preds[i].replace("<pad>", "").replace("[eoo]", "").replace("[cls]", "").replace(
            "[boo]", "")
        real_val_list[i] = real_val_list[i].replace("<pad>", "").replace("[eoo]", "").replace("[cls]", "").replace(
            "[boo]", "")
        if "[ste]" in list_preds[i] and "[ste]" in real_val_list[i]:
            reference = [list_preds[i].split("[ste]")[1].strip().split(" ")]
            candidate = real_val_list[i].split("[ste]")[1].strip().split(" ")
            twogram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            threegram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0),
                                      smoothing_function=SmoothingFunction(epsilon=1e-12).method1)
            total_val.append((twogram + threegram) / 2)
    # 2 and 3 gram mean for all the data
    return np.mean(total_val)


def structural_acc(merged_df_preds):
    total_values = []
    # TODO: add in a check to make sure the prediction is correct before adding it to total_values
    for i in range(0, len(merged_df_preds)):
        if (i % 1000 == 0):
            print(i)
        # find all tokens between square brackets
        tokens_input = re.findall("(?<=\[)[^]]+(?=\])", merged_df_preds["output"].tolist()[i])
        tokens_generated = re.findall("(?<=\[)[^]]+(?=\])", merged_df_preds["generated"].tolist()[i])
        # compare the arrays elementwise to determine number of tokens correctly preserved, divide by total amount of tokens preserved
        try:
            total_values.append((np.array(tokens_input) == np.array(tokens_generated)).sum() / len(tokens_input))
        # short term solution to deal with arrays that are not the same length: talk in meeting about best way to mitigate
        except:
            total_values.append(0);
    return (np.mean(total_values))


def offYAcc(df):

    return sum((("OffY" in row['output'] and "OffY" in row['generated']) or
                ("OffN" in row['output'] and "OffN" in row["generated"]))
               for index, row in df.iterrows()) / len(df)

# Qualitative Testing
def group_testing(df):
    match = []
    for index, row in df.iterrows():
        output_str = row["output"]
        generated_str = row["generated"]
        if (output_str is not None and generated_str is not None):
            output = re.search(r'[grp](.*?)[ste]', output_str)#.group(1)
            generated = re.search(r'[grp](.*?)[ste]', generated_str)#.group(1)
            if (output is not None and generated is not None):
                if (("OffY" in row['output'] and "OffY" in row['generated']) or
                        ("OffN" in row['output'] and "OffN" in row["generated"])):
                    match.append(any((output.group(1) in generate or generate in output.group(1)) for generate in generated.group(1).split()))
            else:
                match.append(0)
        else:
            match.append(0)

    return sum(match) / len(match)

def main(input_predictions, output):
    df = input_predictions.copy()
    if (len(list(df.columns)) == 3):
        df.columns = ["output", "generated", "actual"]
    else:
        df.columns = ["output", "generated"]
    struc_acc = structural_acc(df)
    bleu_score = BLEUScores(df)
    offy_acc = offYAcc(df)
    group_test = group_testing(df)
    acc_df = pd.DataFrame([struc_acc, offy_acc, group_test, bleu_score]).transpose()
    acc_df.columns = ["stuctural_acc", "raw_acc_OffYN", "raw_acc_group", "BLEU_acc"]
    acc_df.to_csv(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-training", nargs="+", help="comma separated file with an input and output row")
    parser.add_argument("--input-validation", nargs="+", help="comma separated file with an input and output row")
    parser.add_argument("--predictions-initial", nargs="+", help="inital predictions dataframe to get an idea of the model")
    parser.add_argument("--output", nargs="+", help="the file to save the model into")

    args = parser.parse_args()

    samples = ["100K"]
    j = 0
    for sample in args.input_training:
        dataset = load_dataset('csv', index_col = 0, data_files={'train': [args.input_training],
                                                  'test': [args.input_validation]})
        train_dataset = dataset['train']
        val_dataset = dataset['test']

        # gpt2: 12
        # gpt2 - medium: 24
        # gpt2 - large: 36
        # gpt2 - xl: 48
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()


        def tokenize(batch):
            tokenized_input = tokenizer(batch['input'], truncation=True, max_length=128, padding='max_length',
                                        add_special_tokens=False)
            tokenized_label = tokenizer(batch['output'], truncation=True, max_length=64, padding='max_length',
                                        add_special_tokens=False)

            #tokenized_input['labels'] = tokenized_label['input_ids']

            return tokenized_input

        special_tokens_dict = {
            'additional_special_tokens': ['[boi]', '[eoi]', '[OffY]', '[OffN]', '[ind]', '[grp]', '[ste]', '[boo]', '[eoo]']}
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


        train_dataset = train_dataset.map(tokenize, batched=True, batch_size=16)
        val_dataset = val_dataset.map(tokenize, batched=True, batch_size=16)

        train_dataset.set_format('numpy', columns=['input_ids', 'attention_mask'])
        val_dataset.set_format('numpy', columns=['input_ids', 'attention_mask'])

        train_dataset = train_dataset.remove_columns(["input", "output"])
        val_dataset = val_dataset.remove_columns(["input", "output"])

        train_dataset = train_dataset.remove_columns(["__index_level_0__"])
        val_dataset = val_dataset.remove_columns(["__index_level_0__"])

        train_dataset.set_format("torch")
        val_dataset.set_format("torch")

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(val_dataset, batch_size=8)

        import os

        i = 0
        for learning_rate in [7e-3]:#, 5e-3, 6e-3]:#[3e-3, 5e-3]:#, 3e-2]:
            for epochs in [20]:#5, 10, 15, 20]:#[10, 20, 50]:#,100,200,500]:
                model = GPT2LMHeadModel.from_pretrained("gpt2")
                model.resize_token_embeddings(len(tokenizer))

                print(str(epochs) + " epochs\n")

                optimizer = AdamW(model.parameters(), lr=learning_rate)

                num_epochs = epochs
                num_training_steps = num_epochs * len(train_dataloader)
                lr_scheduler = get_scheduler(
                    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                )

                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model.to(device)

                progress_bar = tqdm(range(num_training_steps))
                # later: (if model is better than prev, save)
                # set the epochs to 30, and set LR to 0.004 (for larger models) use all data, prioritize this model
                # try 0.003, 0.005 later if I have GPUs
                model.train()
                for epoch in range(num_epochs):
                    output_df = pd.DataFrame(columns=["Source Text", "Generated Text"])

                    for batch in train_dataloader:
                        b_input_ids = batch["input_ids"].to(device)
                        b_labels = batch["input_ids"].to(device)
                        b_masks = batch["attention_mask"].to(device)

                        #batch = {k: v.to(device) for k, v in batch.items()}
                        model.zero_grad()
                        outputs = model( b_input_ids,
                          labels=b_labels,
                          attention_mask = b_masks,
                          token_type_ids=None
                        )
                        loss = outputs[0]
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                        optimizer.zero_grad()
                        progress_bar.update(1)

                        logits = outputs.logits
                        predictions = tokenizer.batch_decode(torch.argmax(logits, dim=-1), padding='max_length', skip_special_tokens=False)
                        references = tokenizer.batch_decode(batch["input_ids"], padding='max_length', skip_special_tokens=False)
                        output_df = output_df.append(pd.DataFrame({'Source Text': references,
                                                                   'Generated Text': predictions}), ignore_index=True)
                    import os
                    path = "model_gpt2_1/epoch_" + str(epoch) + "/model_" + samples[j] + "_" + str(
                        learning_rate) + "_" + str(
                        epochs)

                    # Check whether the specified path exists or not
                    isExist = os.path.exists(path)
                    if not isExist:
                        # Create a new directory because it does not exist
                        os.makedirs(path)
                        os.makedirs(path + "/predictions")
                    isExist = os.path.exists("../model_gpt2_1/final/predictions")
                    if not isExist:
                        os.makedirs("../model_gpt2_1/final/predictions")

                    output_df.to_csv(path + "/predictions/predictions_train.csv")
                    main(output_df, path + "/predictions/predictions_accuracies_train.csv")
                    model.eval()
                    output_df = pd.DataFrame(columns=["Source Text", "Generated Text", "Actual Quote"])
                    for batch in eval_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        inputs = tokenizer(batch, add_special_tokens=False, padding='max_length', return_tensors="pt").to(
                           device)
                        inputs = batch
                        output_sequences = model.to(device).generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            do_sample=False,  # disable sampling to test if batching affects output
                            # eos_token_id = tokenizer.convert_tokens_to_ids("[eoo]"),
                            max_length=64,
                            #repetition_penalty=0.75,
                            # setting ngram repetition penalty to 3 (to be congruent with our similarity score acc. values)
                            #no_repeat_ngram_size=3,
                            # Adding beam search instead of greedy decoding
                            # num_beams = 5,
                            # early_stopping = True
                            # early_stopping = True
                        )
                        source_text = tokenizer.batch_decode(inputs['input_ids'],
                                                padding='max_length', skip_special_tokens=False)
                        #source_text = tokenizer.batch_decode(inputs['labels'],
                        #                        padding='max_length', skip_special_tokens=False)
                        generated_text = tokenizer.batch_decode(output_sequences,
                                                padding='max_length', skip_special_tokens=False)
                        output_df = output_df.append(pd.DataFrame({'Generated Text': generated_text, 'Source Text': source_text}))
                    model.train()
                    path = "model_gpt2_1/epoch_" + str(epoch) + "/model_" + samples[j] + "_" + str(
                        learning_rate) + "_" + str(
                        epochs)

                    # Check whether the specified path exists or not
                    isExist = os.path.exists(path)
                    if not isExist:
                        # Create a new directory because it does not exist
                        os.makedirs(path)
                        os.makedirs(path + "/predictions")
                    isExist = os.path.exists("../model_gpt2_1/final/predictions")
                    if not isExist:
                        os.makedirs("../model_gpt2_1/final/predictions")

                    # save end of each epoch, eval while training
                    # evaluation script for each epoch
                    #embed()
                    output_df.to_csv(path + "/predictions/predictions_dev.csv")
                    main(output_df, path + "/predictions/predictions_accuracies_dev.csv")
                    torch.save(model.state_dict(),
                               path + "/model.pth")

                #TODO: https://huggingface.co/docs/datasets/how_to_metrics
                # make a custom metric
                metric = load_metric("metric_script.py")
                output_df = pd.DataFrame(columns = ["Source Text", "Generated Text"])
                model.eval()
                i = 0
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits

                    predictions = tokenizer.batch_decode(torch.argmax(logits, dim=-1), padding='max_length', skip_special_tokens=False)
                    references = tokenizer.batch_decode(batch["labels"], padding='max_length', skip_special_tokens=False)
                    # TODO: put predictions and references into a dataframe for reference later
                    print([x.replace("<pad>", "") for x in predictions])
                    print("^^predictions")
                    print([x.replace("<pad>", "") for x in references])
                    print("^^those were references")

                    output_df = output_df.append(pd.DataFrame({'Source Text': references,
                                           'Generated Text': predictions }), ignore_index = True)
                    print(output_df)
                    print("number of total batches so far: " + str(i))
                    #embed()
                    metric.add_batch(predictions=predictions, references=references)
                    i = i + 1

                metric.compute()

                output_df.to_csv("model_gpt2_1/final/predictions/model_predictions_grid_search_" + samples[j] + "_" + str(learning_rate) + "_" + str(epochs) + ".csv")
                torch.save(model.state_dict(), "model_gpt2_1/final/model_"+ samples[j] + "_" + str(learning_rate) + "_" + str(epochs) + ".pth")
                #tokenizer.save_pretrained("model_test_4_tokenizer/")
                i = i + 1
    j = j + 1

