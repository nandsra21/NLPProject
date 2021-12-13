from datasets import load_dataset, load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, get_scheduler, AdamW
from transformers.integrations import TensorBoardCallback
from IPython import embed
from torch.utils.data import DataLoader
import tensorflow as tf

# NORMAL DATA!
#dataset = load_dataset('csv', data_files={'train': ['final.tokenized.v2.agg.trn.csv'],
#                                          'test': ['final.tokenized.v2.agg.dev.csv']},
#                       column_names=["input", "output"])

output_dir = 'home/nandsra'

#randomly sample 20 percent of trn data to train the model
dataset = load_dataset('csv', data_files={'train': ['regular.trn.csv'],
                                          'test': ['regular.dev.csv']})
                                          #column_names=["input", "output"])

train_dataset = dataset['train']
val_dataset = dataset['test']

# retrain model with default padding options, see if it works
# reload prediction code and find output

tokenizer = T5Tokenizer.from_pretrained("t5-small")
#tokenizer.padding_side = "left"
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def tokenize(batch):
    tokenized_input = tokenizer(batch['input'], padding='max_length', truncation=True, add_special_tokens=False)
    tokenized_label = tokenizer(batch['output'], padding='max_length', truncation=True, add_special_tokens=False)

    tokenized_input['labels'] = tokenized_label['input_ids']

    return tokenized_input

special_tokens_dict = {'additional_special_tokens': ['[boi]', '[eoi]', '[OffY]', '[OffN]', '[ind]', '[grp]', '[ste]', '[boo]', '[eoo]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(output_dir + "/model_regular_test/tokenizer/")

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=128)
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=128)

train_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])

print(str(train_dataset[0])[:100])
embed()
#print(train_dataset[0]["labels"])

# TODO: need more code to save the model/log the accuracy - look at huggingface

# TODO: print evaluation accuracy (write the log file to a text file that I can open)
# TENSORBOARD - hugging face, and how to use with my code (interface can interactively get your training accuracy per iteration etc)
# checkpoints - save all of them
# create variable for global accuracy
# write conditions break build if the accuracy hasn't improved in n iterations
# batch size and learning rate (play with hyperparameters)
# add scheduling learning rate
# save best 10 models, record all eval accuracies in log, plot them (x: iterations y: dev set accuracy)
# or use Tensorboard to do this (figure it out)

# try to feed input into the model and see what it outputs
# loop through dev set, get predictions for all prompts inthe dev set, compare models predictions with the actual labels (pick a few representative ones and send to channel)

optimizer = AdamW(model.parameters(), lr=5e-5)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=1,  # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
    prediction_loss_only=True,
    # If I need co compute only loss and not other metrics, setting this to true will use less RAM
    learning_rate=1e-5, # this is already a learning rate scheduler " Will default to an instance of AdamW on your model and a scheduler given by get_linear_schedule_with_warmup() controlled by args"
    evaluation_strategy='steps',  # Run evaluation every eval_steps
    save_steps=1000,  # How often to save a checkpoint
    save_total_limit=10,  # Number of maximum checkpoints to save # TODO: look at documentation to figure out how to save checkpoint with highest accuracy (script later on?)
    load_best_model_at_end=True,
    remove_unused_columns=True,  # Removes useless columns from the dataset
    run_name='run_name',  # Wandb run name
    logging_dir= output_dir + '/logs',
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
trainer.save_model(output_dir + '/model_regular_test')

