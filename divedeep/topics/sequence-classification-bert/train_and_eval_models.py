#!/usr/bin/env python
# coding: utf-8

# # Train and eval models

# ## Load dataset

# In[39]:


import matplotlib.pyplot as plt
import itertools
from collections import Counter, OrderedDict
from pprint import pprint


# In[40]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[41]:


import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments
)


# In[42]:


dataset_id = "ag_news"
dataset = load_dataset(dataset_id)
dataset


# Since the dataset does not have an eval split, we generate it from the training set

# In[43]:


split_train_ds = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_train_ds["train"],
    "eval": split_train_ds["test"],
    "test": dataset["test"]
})
dataset


# In[44]:


label_class = dataset["train"].features["label"]
label_names = label_class.names
num_labels = label_class.num_classes
print(f"{num_labels} labels: {label_names}")


# In[45]:


label_class


# In[46]:


encoded_intents = {k: v for v, k in enumerate(label_names)}
inverse_encoded_intents = {k: v for v, k in encoded_intents.items()}


# In[47]:


encoded_intents


# In[48]:


inverse_encoded_intents


# ## Quick check to verify the dataset is not corrupted

# In[49]:


dataset.set_format(type="pandas")


# In[50]:


train_df = dataset["train"][:]
train_df["label_name"] = train_df["label"].apply(label_class.int2str)


# In[51]:


test_df = dataset["test"][:]
test_df["label_name"] = test_df["label"].apply(label_class.int2str)


# In[52]:


train_df.sample(5)


# In[53]:


test_df.sample(5)


# In[54]:


train_df["label_name"].value_counts()


# In[55]:


test_df["label_name"].value_counts()


# In[56]:


dataset.reset_format()


# ## Split and tokenize dataset

# In[57]:


model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# In[58]:


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# In[59]:


tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=None)


# In[60]:


# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# In[61]:


tokenized_dataset["train"][0]


# In[62]:


tokenized_dataset["test"][0]


# ## Define metrics functions

# In[63]:


from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt


# In[64]:


def plot_confusion_matrix(y_true, y_pred, labels):
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(len(labels),len(labels)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(cmap="Greens", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
    
    print(cm)
    return cm


# In[65]:


y_true = [1]*50 + [0]*50
y_pred = [1]*40 + [0]*60


# In[66]:


cm = plot_confusion_matrix(y_true, y_pred, labels=["1", "0"])


# In[67]:


precision_recall_fscore_support(y_true, y_pred)


# In[68]:


def plot_prfs(y_true, y_pred, labels):
    # fig, ax = plt.subplots(figsize=(4, len(labels)))
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

    prfs = precision_recall_fscore_support(y_true, y_pred)
    plt.table(
        cellText=prfs,
        rowLabels=["Precision", "Recall", "F1", "Support"],
        cellLoc="left",
        colLabels=labels,
        colWidths = [.3]*len(prfs[0]),
        loc="center",
        cellColours=plt.cm.hot(prfs),
        bbox=None
    )
    # display.plot(cmap="Greens", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Precision, recall, f1 score, true labels per class")
    plt.show()
    return prfs


# In[69]:


prfs = plot_prfs(y_true, y_pred, labels=["1", "0"])


# In[70]:


# For training loop
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    return {"f1": f1_score(labels, preds, average="weighted")}


# ## Training

# In[71]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[72]:


config = AutoModelForSequenceClassification.from_pretrained(model_id)
config


# In[73]:


model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=encoded_intents,
    id2label=inverse_encoded_intents
)


# In[74]:


state_dict = model.state_dict()


# In[75]:


type(state_dict)


# In[76]:


state_dict.keys()


# In[77]:


for key, values in state_dict.items():
    print(values.shape)
    break


# In[78]:


means = []
stds = []
maxs = []
mins = []

keys_split = OrderedDict()
for key, value in state_dict.items():
    key_dot_split = key.split(".")
    if key_dot_split[1] == "embeddings":
        split_key = "embeddings"
    
    elif key_dot_split[1] == "encoder":
        # eg: key == "bert.encoder.layer.0.attention.self.query.weight"
        # ->: split_key == "econder_layer_0"
        split_key = "_".join(key_dot_split[1:4])
    
    elif key_dot_split[1] == "pooler":
        split_key = "pooler"
    
    elif key_dot_split[0] == "classifier":
        split_key = "classifier"
    
    else:
        raise ValueError(f"Unexpected key: {key}")
    
    value = value.to(torch.float)
    # here value from 0 to embedding dim - 1 
    if key.endswith("position_ids"):
        mean = 0
        std = 0
        max_ = 0
        min_ = 0
    else:
        mean = torch.mean(torch.flatten(value))
        std = torch.std(value)
        max_ = torch.max(value)
        min_ = torch.min(value)
        means.append(mean)
        stds.append(std)
        maxs.append(max_)
        mins.append(min_)
        
    tuple_ = (key, value.shape, max_, min_, mean, std)
    try:
        keys_split[split_key].append(tuple_)
        
    except KeyError:
        keys_split[split_key] = [tuple_]


# In[79]:


def print_model_info_dict(obj):
    for key, value in obj.items():
        print(key)
        pprint(value)
        print("\n" + "="*50 + "\n")

print_model_info_dict(keys_split)


# In[80]:


def max_min_mean_std(obj, name=""):
    print("Stats: ", name)
    print("Mean: ", torch.mean(torch.Tensor(obj)))
    print("Std: ", torch.std(torch.Tensor(obj)))
    print("Max: ", torch.max(torch.Tensor(obj)))
    print("Min: ", torch.min(torch.Tensor(obj)))
    print("\n" + "="*50 + "\n")

# They exclude values from embedding_ids € [0,512)
max_min_mean_std(means, name="mean")
max_min_mean_std(stds, name="std")
max_min_mean_std(maxs, name="max")
max_min_mean_std(mins, name="min")


# In[81]:


model_name = f"{model_id}-finetuned-{dataset_id}"
batch_size = 64
logging_steps = dataset["train"].num_rows // batch_size
train_epochs = 3
lr = 5e-5
weight_decay=0.1
eval_strategy="epoch"

training_args = TrainingArguments(
    output_dir=model_name,
    learning_rate=lr,
    weight_decay=weight_decay,
    evaluation_strategy=eval_strategy,
    logging_steps=logging_steps,
    push_to_hub=False,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)


# In[82]:


trainer = Trainer(
    model=model.to(device),
    args=training_args,
    # data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
    train_dataset=tokenized_dataset["train"].select(range(100)),
    eval_dataset=tokenized_dataset["eval"].select(range(100)),
    tokenizer=tokenizer
)


# In[83]:


trainer.train()


# In[ ]:


preds = trainer.predict(tokenized_dataset["test"].select(range(10)))
preds.metrics


# In[ ]:


y_pred = np.argmax(preds.predictions, axis=1)
y_true = dataset["test"]["label"]


# In[ ]:


cm = plot_confusion_matrix(y_true, y_pred, labels=label_names)


# In[ ]:


prfs = plot_prfs(y_true, y_pred, labels=label_names)

