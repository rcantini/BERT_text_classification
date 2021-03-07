# BERT_text_classification

How to fine-tune a BERT classifier using Huggingface and Keras+Tensorflow.

Two different classification problems are addressed:
- <a href="https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">IMDB sentiment analysis</a>: detect the sentiment of a movie review, classifying it according to its polarity, i.e. *negative* or *positive*.
- <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data">Toxic comment classification</a>: determine the toxicity of a Wikipedia comment, predicting a probability for each type of toxicity, i.e. *toxic*, *severe toxic*, *obscene*, *threat*, *insult* and *identity hate*.

The develeped applications are comprised of the following steps:
- ***Sampling of the dataset***: in order to keep the training times acceptable, both datasets have been sampled.
- ***Fine tuning of the BERT classifier***: a classification layer is stacked on top of the BERT encoder and the entire model is fine-tuned.
- ***Performance evaluation***: I evaluated the trained model using ROC_AUC and Accuracy metrics. I further tested the model by preparing a bunch of sentences, achieving very good results.

For further information, check out my blog post: https://riccardo-cantini.netlify.app/post/bert_text_classification/

## Prediction examples

### Sentiment analysis
<img src="../../review_sentiment/results/pred_sent.PNG" style="display: block; margin-left: auto; margin-right: auto; width: 90%; height: 90%"/>

### Toxicity detection
<img src="../../toxic_comments/results/pred_tox.PNG" style="display: block; margin-left: auto; margin-right: auto; width: 90%; height: 90%"/>

## Requirements
- tensorflow_addons==0.12.0
- matplotlib==3.3.3
- tensorflow==2.4.0
- numpy==1.19.5
- keras==2.4.3
- pandas==1.2.3
- scikit_learn==0.24.1
- transformers==4.3.3
