from datasets import load_dataset, load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, get_scheduler, AdamW
from transformers.integrations import TensorBoardCallback
from IPython import embed
import numpy as np

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-training", help="comma separated file with an input and output row")
    parser.add_argument("--input-validation", help="comma separated file with an input and output row")
    parser.add_argument("--output", help="the file to save the model into")

    args = parser.parse_args()

    dataset = load_dataset('csv', data_files={'train': [args.input_training],
                                              'test': [args.input_validation]})
                                              #column_names=["input", "output"])
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def tokenize(batch):
        tokenized_input = tokenizer(batch['input'], padding='max_length', truncation=True, add_special_tokens=False)
        tokenized_label = tokenizer(batch['output'], padding='max_length', truncation=True, add_special_tokens=False)

        tokenized_input['labels'] = tokenized_label['input_ids']

        return tokenized_input

    special_tokens_dict = {'additional_special_tokens': ['[boi]', '[eoi]', '[OffY]', '[OffN]', '[ind]', '[grp]', '[ste]', '[boo]', '[eoo]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(args.output + "/tokenizer/")

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=128)
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=128)

    train_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])

    print(str(train_dataset[0])[:100])
    embed()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,  # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
        prediction_loss_only=True,
        # If I need co compute only loss and not other metrics, setting this to true will use less RAM
        learning_rate=3e-4, #this is already a learning rate scheduler
        # Will default to an instance of AdamW on your model and a scheduler given by
        # get_linear_schedule_with_warmup() controlled by args
        evaluation_strategy='steps',  # Run evaluation every eval_steps
        save_steps=1000,  # How often to save a checkpoint
        save_total_limit=10,  # Number of maximum checkpoints to save # TODO: look at documentation to figure out how to save checkpoint with highest accuracy (script later on?)
        load_best_model_at_end=True,
        remove_unused_columns=True,  # Removes useless columns from the dataset
        run_name='run_name',  # Wandb run name
        logging_dir= args.output + '/logs',
        logging_steps=1000,  # How often to log loss to wandb
        eval_steps=1000,  # How often to run evaluation on the val_set
        logging_first_step=False,  # Whether to log also the very first training step to wandb
        metric_for_best_model="loss",  # Use loss to evaluate best model.
        greater_is_better=False  # Best model is the one with the lowest loss, not highest.
    )

    def compute_metrics(eval_preds):
        metric = load_metric("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics,
        callbacks=[TensorBoardCallback]
    )

    trainer.train()
    trainer.save_model(args.output)

