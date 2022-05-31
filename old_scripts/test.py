# -*- coding: utf-8 -*-
# test project
"""HuggingFace.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NTAN50Bm0bnqauLdXIk28MEijlEYqRE9

TODO: 

> Use this notebook to create a pretrained model/token that uses all the skills you learned from part 1 and 2 of Huggingface tutorial. Afterwards, use this notebook for part three for fine-tuning a pretrained model.
"""

#!pip install datasets transformers[sentencepiece]

import torch
import torchvision
from torchvision import transforms, datasets

"""Download test dataset"""

train = datasets.MNIST("", train=True, download=True, 
                       transform = transforms.Compose([transforms.ToTensor()])) #because the output from datasets is not in tensor form by default, you have to transform it into one
test = datasets.MNIST("", train=False, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True) 
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
# Batch size: the amount of samples fed into the network at a time. Usually between 8 and 64, because of memory usage, 
# and to make sure the network picks up on generalizable patterns instead of shortcuts from each sample
# shuffle: determines if the dataset is shuffled. This should always be true (unless good reason why not), 
# because it makes sure the network is grabbing data in a random fashion. In the example of MNIST dataset,
# if it wasn't shuffled, the netowrk would first shift weights to make everything zero, then one, etc. 
# and the network would end by classifying everything it gets as input as a nine.

"""Visualizing the data"""

for data in trainset:
  break;

x, y = data[0][0], data[1][0]

"""Note! When looking at data from the dataloader, because pyTorch by default expects more than one piece of data, there will be 3 dimensions ([1,28,28] in this case) rather than just [28,28]. Use "view()" in order to grab the data out of this tensor. """

import matplotlib.pyplot as plt

plt.imshow(x.view(28,28))
plt.show()

"""Hugging Face Tutorial
- Using Pipeline
- Behind the Pipeline (tokenizers, etc)
"""

# pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I'm so excited to be working with NLP!")

# you can input more than just one string - put it in a list (list of strings)
classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
])

"""There are tons of other classifier pipelines. 
- text generation ("text-generation")
- zero shot classification ("zero-shot-classification")
- fill mask ("fill-mask)
- ner ("ner") -- this returns the person, org, and loc in the input
- question and answer ("question-answering")
- summarization ("summarization")

You can also use "pipeline" and supply your own model.
"""

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

"""Behind the pipeline: What makes up the pipeline?

> The pipeline is made up of

1.   Pre-processing
2.   Model
3.   Post-processing

# Pre Processing
Tokenizers take the text inputs given and turns them into numbers the model can use. 
- Splits input into words
- Maps each token to an integer
- Adds additional inputs for the model

Because the tokenizer must return the same format of data that the model is used to, we use AutoTokenizer's `from_pretrained()` method.
"""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint) # get the format of the tokens of the pretrained model "distill BERT"
# there are a LOT of different ways to split tokens up. Most people go with subword tokenization, where frequent words are not split,
# but rare words are made into subwords. For example, "annoyingly" turns into "annoying" "#ly". 
# This reduces the amount of unknown tokens, memory space, and preserves meaning of words (which character tokenization doesn't)

raw_inputs = [
             "I'm so excited about this, I've been waiting my whole life for it",
             "This is the worst thing I've ever done",
             "It's not my favorite"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# padding:  “padding” out tensors that have a jagged array structure with 0s (or whatever you’d like) to have a rectangular structure that can be read into a network
# truncation: if sequences are larger than the max_length allowed by the model, truncate to the max_length the model can take 
# return_tensors = "pt" - return pyTorch tensors (we're using PyTorch backend instead of tensorflow, but you can choose)
print(inputs)

# attention mask: What is it?
# Because transformers use context clues, this padding to create a rectangular array can throw off answers. Therefore, they must be masked.
# ATTENTION MASK: another tensor (same shape as the input ID) that indicates which tokens should and should not be attended to (which should be ignored by the model)
# 1's are meant to be attended to, 0s should be ignored.
# https://huggingface.co/course/chapter2/6?fw=pt (good overview of how to set up a pretrained system with tokenizer)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# in the format [a, b, c]
# a: batch size
# b: sequence length (with padding)
# c: hidden size (vector dimension of the model input) - these layers cannot be changed by the developer

from transformers import AutoModelForSequenceClassification
# make sure the model used by the AutoTokenizer is the same as the model used by AutoModel. otherwise, the data format may not match

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # make sure to use the AutoModel needed for your purpose.

outputs = model(**inputs)

print(outputs.logits.shape) # logits are the 3 sentences and the 4 labels, from the model. 
print(outputs.logits) # these numbers are not usable for us! We now have to POSTPROCESS them to find out what the model has given us back as output.

"""# Post Processing
- Softmax the tensors (activation function)
"""

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) #we don't know how many dimensions the output is -> dim=-1
print(predictions)

model.config.id2label

# therefore, the first sentence is positive, the second is negative, and the third is negative.

"""# Fine-Tuning a Pretrained Model
- Preparing a large dataset from the Hub
- Using the Trainer API to fine tune the model
- Using a custom training loop
- Using the HuggingFace Accelerate library to easily run the custom training loop on any distributed setup.

([Colab Notebook from Tutorial](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter3/section2_pt.ipynb))
"""

import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Training a sequence classifer (from before) on one batch using PyTorch
checkpoint = "bert-base-uncased" # model name
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # pretrained tokenizer for the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # the model for Sequence Classification
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
] # raw sequence input 
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt") # passed into the tokenizer, where padding is true (rectangular), truncation is true (no longer than max length), with PyTorch tensors

# This is new: this is saying that we have 1 batch, labeled 1
batch["labels"] = torch.tensor([1, 1])

# the type of optimizer. Most people use Adam, but it's worth messing around with depending on what kind of output you want
optimizer = AdamW(model.parameters())
# computing the loss of the model
loss = model(**batch).loss
# backwards prop
loss.backward()
# step function for the optimizer. 
optimizer.step()

# loading in a dataset from the "Hub" for hugging face
# this is an example of the MRPC dataset (https://paperswithcode.com/dataset/mrpc)
# This is one of the 10 datasets composing the GLUE benchmark,
# which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

# the DatasetDict object contains the training, validation, and test set. 
# These each contain their own columns (sentence1, sentence2, label, index), and diff row amounts (# of elements in the set)
raw_train_dataset = raw_datasets["train"] # viewing the first train dictionary sample
raw_train_dataset[0]

# Because the label is already an integer, no preprocessing necessary. After checking raw_train_dataset.features, we see that 
# label is 2 classes, where 0 is not equivalent and 1 is equivalent

"""Preprocessing time! We only have to tokenize sentence1 and sentence2, as the label is already an integer. We do this using `AutoTokenizer.from_pretrained()`. """

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# Okay, almost there. We need to first make these two tokenizer sentences a tuple, so the model can recieve them as a pair. We also need to convert these IDs into tokens.
# In general, you don’t need to worry about whether or not there are token_type_ids in your tokenized inputs: 
# as long as you use the same checkpoint for the tokenizer and the model, everything will be fine as the tokenizer knows what to provide to its model.
# tokenizer.convert_ids_to_tokens(inputs["input_ids"]) -> these are what tell the model what part of the input is the first and what part is the second sentence
# usually not required (because of AutoTokenizers from_pretrained will do it if the model needs it), so make sure tokenizer's model and AutoModel's model line up
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
# took both sentences in to make a tuple, padded and truncated to max length

# Issues with this! 
# 1) returned a dictionary, and only works if we have enough RAM to store the entire dataset. Instead, we can use the map method that 
# takes a dictionary and return a new dictionary with the correctly preprocessed inputs. 
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
# use batched=True so that we can apply the function to multiple elements at once, not separately (which would take a long time)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# You can use multiprocessing when applying your preprocessing function with Dataset.map by passing along a num_proc argument. 
# (not needed when using an AutoTokenizer, because it already uses multiple threads to tokenize the samples)
# Our tokenize_function returns a dictionary with the keys input_ids, attention_mask, 
# and token_type_ids, so those three fields are added to all splits of our dataset.

# The last thing we will need to do is pad all the examples to the length of the longest element when we batch elements together 
# — a technique we refer to as dynamic padding.

"""Dynamic Padding:
> Pad all the examples to the length of the longest element when we batch elements together
"""

# In PyTorch, the function that puts together samples in a batch is the COLLATE function, passed in when you build a DataLoader
# Why did we wait so long to pad? If we pad by batch, we can avoid having extra long padding because of one or two long samples somewhere in the full dataset,
# wasting memory and space.
# COLLATE: puts together all the samples in a batch.
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# sample to batch together (remove anything that is a string), look at length of entry in the batch
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]

"""Dynamic padding means the samples in this batch should all be padded to a length of 67, the maximum length inside the batch. **Without dynamic padding, all of the samples would have to be padded to the maximum length in the whole dataset, or the maximum length the model can accept.**"""

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

# we now have gone from raw text to batches the model can deal with!

"""# Fine-tuning a model with the Trainer API"""

# The Trainer class for HuggingFace transformers helps fine tune pretrained models on the dataset.

# we need to define TrainingArguments, which has all the hyperparameters the Trainer will use for training and eval.
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

#defining a model (pretrained) which we can finetune
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

"""From [HuggingFace](https://huggingface.co/course/chapter3/3?fw=pt):


> You will notice that unlike in Chapter 2, you get a warning after instantiating this pretrained model. This is because BERT has not been pretrained on classifying pairs of sentences, so the head of the pretrained model has been discarded and a new head suitable for sequence classification has been added instead. The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head). It concludes by encouraging you to train the model, which is exactly what we are going to do now.


"""

# We define a Trainer by passing our model, training_args, training and validation, data collator (batches the data), and our tokenizer. 
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
# while the default data_collator in Trainer is the same as the data_collator we defined, it was important to understand. You could delete the line here.

"""From [HuggingFace](https://huggingface.co/course/chapter3/3?fw=pt)
 > `trainer.train()` will start the fine-tuning (which should take a couple of minutes on a GPU) and report the training loss every 500 steps. It won’t, however, tell you how well (or badly) your model is performing. This is because:

1. We didn’t tell the Trainer to evaluate during training by setting evaluation_strategy to either "steps" (evaluate every eval_steps) or "epoch" (evaluate at the end of each epoch).
2. We didn’t provide the Trainer with a compute_metrics function to calculate a metric during said evaluation (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).
"""

trainer.train()
output_dir = '/home/nandsra'
trainer.save_model(output_dir + '/model_test')

#model_dir = output_dir + '/model'

"""# Evaluation
Can we use the performace of the network this time to improve the next time we train?

Making a compute_metrics function!
- takes an EvalPrediction object (tuple with predictions field and label_ids field), returns a dictionary mapping strings to floats. To get predictions, use 
`Trainer.predict()`.


"""

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# the output from the predict function is a tuple with 3 fields: "predictions", "label_ids", and "metrics". 
# metrics: contains loss on dataset, and time metrics. Compute_metrics will add metrics from the function to this section of the tuple.

# predictions is a 2-D array, with the logits from the transformer. 408 is the # of elements we used, and we transform this into predictions about the classification of text (max on second axis).
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

# compare the predictions to the actual labels
from datasets import load_metric
# we will use metrics from the HuggingFace library for the compute_metrics function
metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc") # load metrics to be added for the dataset 
    logits, labels = eval_preds # separate the logits from the labels from the network's output
    predictions = np.argmax(logits, axis=-1) # find the largest value in each output (node with the highest probability)
    return metric.compute(predictions=predictions, references=labels) # return the accuracy (pred vs actual labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch") # we use epoch here and a new model so we don't train on the model we trained earlier
# If you want to fine-tune your model and regularly report the evaluation metrics (for instance at the end of each epoch), use epoch
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics # uses the metrics WE want the trainer to use/compute, not the default. These are the metrics that work better for our problem than the default.
)

trainer.train()

"""# How to train a model without Hugging Face? Using PyTorch instead"""

# summary of what we need (from previous sections, but put together)
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc") #loading data
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) #loading pretrained tokenizer

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True) #using the tuples so the network recognizes two sentences as a pair

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) # map the dataset to the tokenizer version using the function we wrote
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # batch the dataset (arranging dataset)

"""`Trainer` does a lot of postprocessing for us that we need to do on our own without it. In particular, 
1. remove columns that the model does not expect (sentence1, sentence2, which are strings)
2. rename column label to labels (what the model expects)
3. set format of datasets to return PyTorch tensors instead of lists


"""

tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# define our dataloaders
from torch.utils.data import DataLoader
# we did this in the section called "loading data from PyTorch" at the top
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# instantiate the model:
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# pass batch to model:
# All 🤗 Transformers models will return the loss when labels are provided, 
# and we also get the logits (two for each input in our batch, so a tensor of size 8 x 2).
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

"""We need 
1. An Optimizer (we use Adam here)
2. A learning Rate Scheduler (linear decay function, to make sure the network doesn't take too big of steps and miss the loss, or too little and end up in a local minimum)
"""

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
# to define the learning rate scheduler, we need
# 1) # of training steps we take
# 2) number of epochs we want to run * # of training batches
num_epochs = 3 # default of Trainer, so we reuse it here
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print(num_training_steps)

"""# The Training Loop"""

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

from tqdm.auto import tqdm # gives a progress bar so we know how much longer training will take

progress_bar = tqdm(range(num_training_steps))

model.train()
# the main part of this loop looks the same as the one from above. Check comments there for meanings of these things.
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()} # switching out the batch 
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad() # make sure gradients don't add up and create exploding gradients
        progress_bar.update(1)

from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): # this makes sure we don't call gradient. If we have gradient, it keeps adding it together (which can cause exploding gradients)
      # for batch training, start at zero, or no gradients unless you don't have space for a full batch on your computer, then zero grad after batch is over.
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"]) # the metrics accumulates batches for us as we go over pred loop with this function
    # is a better option than metric.compute in every section of the for loop, and concatenating it ourselves. 

metric.compute()

"""# Using HuggingFace Accelerate to supercharge your training loop


> Enables distributed training on multiple GPUs and TPUs

"""

# In order to benefit from the speed-up offered by Cloud TPUs, 
 # we recommend padding your samples to a fixed length with the 
 # `padding="max_length"` and `max_length` arguments of the tokenizer.

from accelerate import Accelerator # NEW
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator() # NEW

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# OLD: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# OLD: model.to(device)

# NEW
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
  for batch in train_dataloader:
# OLD:    batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
# OLD:    loss.backward()
          accelerator.backward(loss) #NEW (doing backprop on parallel GPU inits)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)

"""From [HuggingFace](https://huggingface.co/course/chapter3/4?fw=pt)

> The first line to add is the import line. The second line instantiates an `Accelerator` object that will look at the environment and initialize the proper distributed setup. 🤗 Accelerate handles the device placement for you, so you can remove the lines that put the model on the device (or, if you prefer, change them to use `accelerator.device` instead of device).

> Then the main bulk of the work is done in the line that sends the dataloaders, the model, and the optimizer to accelerator.prepare. This will wrap those objects in the proper container to make sure your distributed training works as intended. The remaining changes to make are removing the line that puts the batch on the device (again, if you want to keep this you can just change it to use `accelerator.device`) and replacing `loss.backward()` with `accelerator.backward(loss)`.
"""

# Full and complete training loop using HuggingFace's Accelerate

from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

"""From [HuggingFace](https://huggingface.co/course/chapter3/4?fw=pt): 
> Putting this in a train.py script will make that script runnable on any kind of distributed setup. To try it out in your distributed setup, run the command:

```
accelerate config
```

> which will prompt you to answer a few questions and dump your answers in a configuration file used by this command:

```
accelerate launch train.py
```

> which will launch the distributed training.



> If you want to try this in a Notebook (for instance, to test it with TPUs on Colab), just paste the code in a training_function and run a last cell with:

```
from accelerate import notebook_launcher

notebook_launcher(training_function)
```
"""
