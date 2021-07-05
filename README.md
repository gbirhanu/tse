# Tweet Sentiement Mining, Extracting Sentiment supporting phrase from tweet

Tweet sentiment extraction is a featured competition provided by Kaggle. The main task of this competition is, to extract the supporting phrase which helps us to state that the given tweet has positive, negative, or neutral sentiment. For example, if the tweet says "Today's meeting was very boring" the sentiment is negative and the word which makes it negative is "boring". That is what is asked by the competition.

## Challenge of the competition

The main challenge for the competition comes from the nature of the data provided. The data is a tweet from any user from Twitter. And the supporting phrase is part of the original tweet which was selected by crowed workers of Appen company. When users select a supporting phrase they may select it in different ways. Some may include punctuation. Some may ignore punctuation.
So the given task is to select appropriate text from the given tweet which supports the sentiment.

### Text and Selected_text relation

selected_text in the training set is the subset of the text. and the following conditions are met:

- selected_text is a continuous subset (text) from the original tweet. that means for example if the text is "I am bored with today's meeting" the selected_text will be bored or I am bored. But it cannot be like **I bored**. No jumping in the text
- The length of the selected text is less than that of the text except for the neutral tweet.
- The selected text of the neutral tweet has an enormous similarity(Jaccard) with its text.

### Noise in the data.

The data have a lot of weird texts which are not in the natural language. Some of the examples are the following.

- unnecessary Space: The Text has some unnecessary spaces which may be introduced during data cleaning. One example is indicated below.

is back home now gonna miss everyone

## Framing the problem

The next big problem is how to frame the problem itself. I have come across a lot of reading on this competition. I have seen people who try to solve the problem as NER (named entity recognition) by having positive, negative vocabulary as a named entity. The most plausible way of thinking about this problem is as QA(question and answer) problem. The given tweet can be considered as a Question whereas the supporting phrase selected from the text can be considered as an Answer.

## Evalution

According to the official description of the problem, the Evaluation criteria used for scoring the result is Jaccard Similarity. The metric in this competition is the word-level Jaccard score. A good description of Jaccard's similarity for strings is here.

A Python-based implementation code is provided in the competition description.

```
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric, then, is:

![Evaluation Metrics](https://github.com/gbirhanu/tse/blob/main/formula.png?raw=true)

Where :-

---

$n = \textrm{number of documents}$ <br>
$jaccard = \textrm{the function provided above}$ <br>
$gt_i = \textrm{the ith ground truth}$<br>
$dt_i = \textrm{the ith prediction}$

## My solution

After trying a lot of different models I have got a score of 0.73216 on the Private data set and 0.72738 on the Public dataset.
!["Private Score"](https://github.com/gbirhanu/tse/blob/main/private_LB%20score.PNG?raw=true)
My score looks the following
!["My score"](https://github.com/gbirhanu/tse/blob/main/my_private_score.PNG?raw=true)

## Solution Overview

Now comes the question. How and what am I going to solve? Let say the given tweet is "had a very good day and is now going to get into bed!". What we want to know is the supporting phrase or word which tells us the sentiment of this tweet. Here the supporting phrase will be "Very good day". So we just take the phrase from the tweet. As I have mentioned above the supporting phrase will be continuous text in the tweet, so if we know where it starts and where it ends the rest will be obvious. So, our work is to find the probability of each word(token) in the tweet to be the start or the end token of the supporting phrase. I the above example "Very" will have a high probability to be the start token while "day" will have a high probability to be the end token. Then the rest will be inferred.

### Which Machine Learning Algorithm will be good?

The above problem is why we have [masked Languge Modelling](https://arxiv.org/abs/2011.00960). A lot of [transformer](https://arxiv.org/abs/1706.03762) based lanuage model like [BERT](https://arxiv.org/abs/1810.04805), [ROBERTa](https://arxiv.org/abs/1907.11692), [ALBERT](https://arxiv.org/abs/1909.11942),and others are pre-trained based on the masked language model. In MLM, Some tokens (for example 15%) of the dataset will be substituted with the "MASK" token, then the model will be trained to predict that Masked token. By this method, we have a pre-trained model available to us by different people. Further, these models are finetuned for different NLP tasks like question answering, text summarization, and so on. In this work I have selected the RoBERTa model which is trained on [Squad2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15816213.pdf) Question-answering dataset. We have framed the problem as Question answering problem. The given tweet will be treated as a question and the supporting phrase will be treated as an Answer. So from the given tweet what will be the probability of each token for being the start and the end of the supporting phrase? is what we are going to solve.

## What makes my solution unique!

- I used only [Roberta base Squad2](https://arxiv.org/abs/1907.11692) For level one modeling. With the single Roberta Model, I got a score of 0.71421 on the Private Leader board.
- After checking the noise in the data which results from extra spaces in the tweet, I preserve all spaces in the tweet and train it. After doing that I got 0.71729 which add the score by 0.00308.
- I have tried Post-processing suggested and used by some participants of the competition. I got an extra point on my single model by scoring 0.72054. I have to refrain from using post-processing in my final ensemble.
- I have trained my model using [K-Fold](http://statweb.stanford.edu/~tibs/sta306bfiles/cvwrong.pdf) sampling technique. I used value k= 5, so for each fold training, I used single fold as validation data while training the model on the remaining four samples. I have also saved out-of-fold prediction (OOF) to refer to in my second-level model.
- I have used five different seeds and sample my data and train all five different samplings. So my first level model is the number of seed * the number of fold = 5*5 = 25 models. I have used Google Colab pro to train them since the Kaggle platform is not enough.
- From first- level model I got two outputs
  - Character level out of fold prediction:- When I use 5-fold sampling I will train my data using four samples and make the first sample as validation data. That means I try to predict the probability of the given token in the tweet for being starting and ending tokens for the selected_text (supporting phrase) in this sample. And I saved it for the second-level model.
  - Character level Test dataset probability prediction After training is completed

### Level Two modeling

I used [Stack Ensembling](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) technique to combine the first level model which is RoBERTa with different seeds. Here the input to this model is character level start and end probabilities of a given text. Then the data will be retrained to adjust those probabilities. For this, I used RNN(LSTM) model. The result is significantly increased as depicted in the above picture.
In general, the model looks like the following:<br>
!["my Model"](https://github.com/gbirhanu/tse/blob/main/model.PNG?raw=true)

## Training:

LEVEL ONE MODEL <br>
As I have mentioned above I have trained my level one training using the RoBERTa model with five different seeds. For each model with each seed, I have used Stratified K-fold sampling. The 'k' in my case is five. So, I have five folds of data. for each fold, I have trained it for 4 EPOCH. Batch size = 32, MAX Sequence length 106.

To speed up training I have used [Adam optimizer with linear warmup scheduler](https://huggingface.co/transformers/main_classes/optimizer_schedules.html). This will help the model to adjust the learning rate. <br>

LEVEL TWO MODELLING:<br>
For lever two modeling I have used the RNN model. this model accepts as input character level probability prediction. I have trained it for 20 epochs.

LOSS:

For the first-level model, KLDivloss was used while I have used Standart cross-entropy loss for the second-level model. I have added [label smoothing](https://paperswithcode.com/method/label-smoothing) which increases accuracy.

## Final Inference

The final Inference is based on the output of my model from the level two model. I have selected my start and end tokens of supporting phrases using the SoftMax function. the softmax function gives me the token with maximum probabilities of to be the start and end token of the selected_text(supporting Phrase).
