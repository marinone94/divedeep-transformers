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
Imagine you are the director of The Only Truth, a fictional American digital newspaper from Kentucky, and you want to monitor which types of articles generate more interactions. Sure, you could ask your editors to tag the articles according to a set of categories, but what if you want to know how this behavior evolved in the last ten years? Yeah, someone could manually do that. But what if you want to get a general picture including also your competitors? That is a tipical case where text classification comes in help.

We will use this use case to learn the details of the transformers architecture, infer specific and general rules about BERT, and to learn how to

## Huggingace, BertViz, Ecco, ...
Take parts of the intro

## AG News
Link to both Hugging Face datasets, source, and paper.

## BERT
Intro to the model. Intro to the implementation.

## F1-score. That's it?