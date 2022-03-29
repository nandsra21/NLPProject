from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from datasets import load_metric

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-training", help="comma separated file with an input and output row")
    parser.add_argument("--input-validation", help="comma separated file with an input and output row")
    parser.add_argument("--output", help="the file to save the model into")

    args = parser.parse_args()

    dataset = load_dataset('csv', index_col = 0, data_files={'train': [args.input_training],
                                              'test': [args.input_validation]})
    # column_names=["input", "output"])
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # take from most recent checkpoints
    model = T5ForConditionalGeneration.from_pretrained("t5-small")  # "model_4/checkpoint-110000")#"t5-base")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()


    def tokenize(batch):
        tokenized_input = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=128,
                                    add_special_tokens=False)
        tokenized_label = tokenizer(batch['output'], padding='max_length', truncation=True, max_length=64,
                                    add_special_tokens=False)

        tokenized_input['labels'] = tokenized_label['input_ids']

        return tokenized_input


    special_tokens_dict = {
        'additional_special_tokens': ['[boi]', '[eoi]', '[OffY]', '[OffN]', '[ind]', '[grp]', '[ste]', '[boo]', '[eoo]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(args.output + "/tokenizer/")

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=32)
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=32)

    train_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = train_dataset.remove_columns(["input", "output"])
    val_dataset = val_dataset.remove_columns(["input", "output"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=3e-4)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,0 ,num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    #TODO: https://huggingface.co/docs/datasets/how_to_metrics
    # make a custom metric
    metric = load_metric("metric_script.py")

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

