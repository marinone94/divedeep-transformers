# CATCHYNAME - Sequence classification with BERT

# Table of content
* [Introduction](#introduction)
    * [The task](#news-topic-classification)
    * [Root repos](#huggingace-bertviz-ecco)
    * [The dataset](#ag-news)
    * [The model](#bert)
    * [The metrics](#f1-score-thats-it)
* Architecture
    * Paper
    * Implementation
    * Full visualization (interactive with Javascript?)
* Pretrained model
    * Masked-tokens predictions
    * Word similarity
* Fine-tuning
    * Tokenization
    * Metrics evolution
    * Resource utilization
* Inference
    * Features importance
    * POS / NER impact
    * 
* How can we improve it?
* Real-world applications
* Next CATCHYNAME

# Introduction
Extracting a summary, determining sentiment, classifying the needs of a customer, determining if an email is spam or a phsihing attempt, categorizing documents, service desk tickets or claims, finding hate-speech on social media... the list of real world applications for text-based sequence classification is endless.

As in any problem, it is a good rule to start defining what we will be working with. In general, sequence classification is the task of predicting a category (label) given a sequence of inputs. Often, within the Natural Language Processing domain, sequence classification applied to text inputs is referred to as text classification. However, some people place under the same umbrella tasks where one wants to classify every word in a sentence, or every word in a document, for example finding personally identifiable information (PII) in a document. Usually, those tasks are rather tagged as token classification ones, but that is not always the case. I like this distinction as it is intuitive, therefore it will be adopted here and in all the other contents, unless specified.

In this article, we will focus on the task of predicting the overall category of a sequence of words, either it being a sentence or a set of sentences. We will primarily focus on the [BERT](https://arxiv.org/abs/1810.04805) model, an encoder-only transformer model which can be adapted and trained for text classification. Specifically, we will fine-tune a pretrained BERT model using Hugging Face [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), and [Datasets](https://github.com/huggingface/datasets) libraries. To grasp all the details, we will experiment with different datasets, leverage [BertViz](https://github.com/jessevig/bertviz) and [Ecco](https://github.com/jalammar/ecco) libraries, and edit the source codes ourselves. We will compare how the models diverge when fine-tuned on different datasets, understand why they fail on specific examples, and try to infer general rules to improve the models performance.

As in all the CATCHYNAME articles, I assume the readers have some experince in Python and grasp the fundamentals of Machine Learning. I hope you will enjoy this article, as well as the future ones.

Happy reading. Go deep. Do good.

## News topic classification
Imagine you are the director of The Only Truth, a fictional American digital newspaper from Kentucky, and you want to monitor which types of articles generate more interactions. Sure, you could ask your editors to tag the articles according to a set of categories, but what if you want to know how this behavior evolved in the last ten years? Yeah, someone could manually do that. But what if you want to get a general picture including also your competitors? Manual tagging can quickly reach an unfeasible scale. That is a tipical case where text classification comes in help.

We will use this use case to learn the details of the transformers architecture, infer specific and general rules about BERT, and to learn how to ... TBC

## Huggingace, BertViz, Ecco, ...
Take parts of the intro TBC 

## AG News
Link to both Hugging Face datasets, source, and paper.

## BERT
Intro to the model. Intro to the implementation.

## F1-score. That's it?

# Bidirectional Encoder Representations from Transformers (a.k.a. BERT)
BERT is an encoder-only transformer designed to learn deep representations from text. But what does this mean? Long story short, any textual information needs to be transformed into a squence of numbers, which can then be processed by Machine Learning algorithms. Those arrays are called embeddings. The better words and sentences are represented by their embeddings, the more accurate the predictions will be, and BERT does exactly that.

The model leverages several innovative ideas introduced in the last decade, and their joint use allowed BERT to outperform any other model in almost any NLP task by a large margin. The three most important concepts are **transfer-learning**, **self-supervised learning**, and **attention**. 

Transfer learning is the basis of human learning. It simply means to leverage previous knowledge to learn new things in a much more effective way. Imagine if you had to learn to walk every time you wanted to try a new sport... The same concept is nowadays applied to most of the Machine Learning task. BERT, and all the transformers-based models, leverage a simple but powerful idea: they are split into a body and a head, which have different structure and goals. To make it (overly) simple, the body "learns the language" and the head "learns the task".

The body's parameters are trained to build the best possible embeddings in a language (pre-training), and they are reused for multiple tasks. This means the pretraining is done once and reused many times, making it "conventient" (economically valuable) to pretrain large models on huge datasets.
Although the main goal of pretraining the body of a model is to learn a general representation of the language so that it can then be tuned on specific tasks (fine-tuning), a "task" needs to be used in the process. FIND A GOOD WAY TO SAY SUPERVISED LEARNING IS USED/NNEDED AND ACHIEVE BETTER PERFORMANCE THAN UNSUPERVISED. However, manually labelling large amount of data is prohibitelty expensive. But wait, are we sure we cannot somehow automate this? No, we are not.

Self-supervised learning is essential to leverage the massive amount of unlabeled data available nowadays. It simply means to mask some information in the data, and train the model to predict it. This can be done in several ways, but the most common ones in NLP are to:
* mask some tokens and ask the models to predict them, using context from both the left and the right.
* mask all the future tokens and ask the model to predict them, using only context from the left.
* take two sentences and ask the model to predict whether they are consecutive in the source dataset.
The first and second tasks are exclusive, and the choice dpended mainly whether the goal was to get a generative or a discriminative model. 
In eithr case, the direct consequence of learning those "general" tasks is that the model learns how to build an accurate embedding representation of the language which can be used for many other tasks like text classification, text generation, question answering, summarization, translation, etc. It is enough to replace the head with a new one, fine-tune the model on a specific (usually much smaller) dataset, and the model will learn the task much quicker and deliver much better results. BERT leverages the first and last pretraining tasks, since the goal was to pretrain the body for then performing discriminative tasks.

The last key concept to introduce is attention. Attention is a powerful mechanism that allows the model to learn to focus on a specific part of the input, and to ignore the rest. This is the key to BERT's success.

## 