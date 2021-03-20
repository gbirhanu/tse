# Tweet Sentiement Mining, Extracting Sentiment supporting phrase from tweet

Tweet sentiement extraction is featured competition provided by Kaggle. The main task of the competition is to extract the supporting phrase which helps us to call some tweet as positive, negative or neutral. For example if the tweet says "Todays meeting was very boring" the sentiment is negative and the word which makes it negative is "boring". That is what is asked by the competition.

## Challenge of the competition

The main challenge for the competition comes from the nature of the data provided. the data is a tweet from any user from twitter. And the supporting phrase is part of the original tweet which was selected by crowed workers of Appen company. When user select supporting phrase they may select in diffrent ways. Some may include panctuation. some may ignore panctuation.
So the given task is to select appropriate text from the given tweet which actually support the sentiment.

### Text and Selected_text relation

selected_text in the training set is the subset of the text. and the following conditions are met:

- selected_text is continues subset (text) from the original tweet. that means for example if the text is "I am bored with todays meeting" the selected_text will be bored or I am bored. But it cannot be like **I bored**. No jumping in the text
- the length of the selected text is less that that of text except for neutral tweet.
- selected text of the neutral tweet have very big similarity(jaccard) with its text.

### Noise in the data.

The data have a lot of weird texts with is not in the natural language. Some of the examples are the following.

- unnecessary Space: The Text have some unnecessary spaces which may be injected during data cleaning. One example is shown blow.

is back home now gonna miss every one

## Framing the problem

The next big problem is how to frame the problem it self. I have come across a lot of reading on this competition. I have seen people who try to solve the problem as NER (named entity recognition) by having postitive, negative vocabulary as named entity. The most plausable way of thinking about this problem is as QA(question and answer) problem. The given tweet can be considered as Question and the supporting phrase selected from the text can be considered as an Answer.

## Evalution

According to the official description of the problem the Evaluation criteria used for scoring the result is by using Jaccard Similarity. The metric in this competition is the word-level Jaccard score. A good description of Jaccard similarity for strings is here.

A Python implementation based code is also given in the compitition description.

```
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric, then, is:

$score = \frac{1}{n} \sum_{i=1}^n jaccard( gt_i, dt_i )$

Where :-

---

$n = \textrm{number of documents}$ <br>
$jaccard = \textrm{the function provided above}$ <br>
$gt_i = \textrm{the ith ground truth}$<br>
$dt_i = \textrm{the ith prediction}$

## My solution

After trying a lot of diffrent models I have got score of 0.73216 on Private data set and 0.72738 on Public dataset.

!["Private Score"]("private_LB score.PNG", "Leader board for private Dataset")
My score looks the following
!["My score"]("my_private_score.PNG", "my score")

## Solution Overview

Now comes the question. How and what i am going to solve? Let say the tweet given is "had a very good day and is now going to get into bed!". What we want to know is the supporting phrase or word which tells us the the sentiment of the tweet. Here the supporting phrase will be "Very good day". So we just take the phrase from the tweet. As I have mentioned above the supporting phrase will be continous text in the tweet, so if we know where it starts and where it ends the rest will be obvious. So, our work is to find the probablity of each word(token) in the tweet to be the start or the end token of the supporing phrase. I the bove example "Very" will have high probablity to be the start token while "day" will have high probality to be the end token. Then the rest will be infered.

### Which Machine Learning Algorithm will be good?

The above problem is why we have [masked Languge Modelling](https://arxiv.org/abs/2011.00960). A lot of [transformer](https://arxiv.org/abs/1706.03762) based lanuage model like [BERT](https://arxiv.org/abs/1810.04805), [ROBERTa](https://arxiv.org/abs/1907.11692), [ALBERT](https://arxiv.org/abs/1909.11942),and others are pretrained based on the masked language model. In MLM, Some token (for example 15%) of the dataset will be substituted with "MASK" token, then the model will be trained to predict that Masked token. By this method we have pretrained model available to us by diffrent people. Further, this models are finetuned for diffrent NLP tasks like Question answering, text summerization and so on. In this work I have selected RoBERTa model wwhich is trained on [Squad2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15816213.pdf) question-answering dataset. We have framed the problem as Question answering problem. The given tweet will be treated as question and the supporting phrase will be treated as Answer. So from the given tweet what will be the probablity of each token for being the start and the end of the supporing phrase? is what we are going to solve.

## What makes my solution unique!

- I used only [Roberta base Squad2](https://arxiv.org/abs/1907.11692) For level one modelling. With single Roberta Model I got score of 0.71421 on Private Leader board.
- After checking the noise in the data which results from extra spaces in the tweet, I preserve all spaces in the tweet and train it. After doing that I got 0.71729 which add the score by 0.00308.
- I have tried Post processing suggested and used by some participant of the competition. I got extra point on my single model by scoring 0.72054. I have refrain from using the post processing in my final ensemble.
- I have trained my model using [K-Fold](http://statweb.stanford.edu/~tibs/sta306bfiles/cvwrong.pdf) sampling techinique. I used value k= 5, so for each fold training, I used single fold as validation data while train the model on remaining four sample. I have also saved out of fold prediction (OOF) to refer in my second level model.
- I have used five diffrent seed and sample my data and train all five diffrent sampling. So my first level model is number of seed * number of fold = 5*5 = 25 models. I have used Google Colab pro to train them since kaggle platform is not enough.
- From first- level model I got two out puts
  - Character level out of fold prediction :- When I use 5-fold sampling i will train my data using four sample and make the first sample as validation data. That means I try to predict the probality of the given token in the tweet for being sarting and ending tokens for the selected_text (supporting phrase) in this sample. And I saved it for the second level model.
  - Character level Test dataset probablity prediction After training is completed

### Level Two modeling

I used [Stack Ensembling](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) techinique to combine the first level model which is RoBERTa with diffrent seed. Here the input to this model is character level start and end probablities of given text. Then the data will be retrained to adjust those probablities. For this I used RNN(LSTM) model. The result is significantly increased as depicted in the above picture.
In general the model looks like the following:<br>
!["my Model"]("https://github.com/gbirhanu/tse/blob/main/model.PNG", "My model")

## Training:

LEVEL ONE MODEL <br>
As I have mentioned above I have trained my level one training using RoBERTa model with five diffrent seed. For each model with each seed I have used Stratified K-fold sampling. The k in my case is five. So, I have five folds of data. for each fold, I have trained it for 4 EPOCH. Batch size = 32, MAX Sequnce length 106.

To speed up training I have used [Adam optimizer with linear warmup scheduler](https://huggingface.co/transformers/main_classes/optimizer_schedules.html). This will help the model to adjust learning rate. <br>

LEVEL TWO MODELLING:<br>
For lever two modelling I have used RNN model. this model accept as input character level probablity prediction. I have trained it for 20 epochs.

LOSS:

For first level model KLDivloss was used while, I have used Standart cross-entropy loss for second level model. I have added [label smoothing](https://paperswithcode.com/method/label-smoothing) which increases accuracy.

## Final Inference

The final Inference is based on the output of my model from level two model. I have selected my start and end tokens of supporting phrase using SoftMax function. the softmax function gives me the token with maximum probablities of to be the start and end token of the selected_text(supporting Phrase).
