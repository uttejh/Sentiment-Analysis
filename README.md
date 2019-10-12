# Sentiment-Analysis
A Recurrent Neural Network to perform Sentiment Analysis on **Movie Reviews**. 

## Built With
- PyTorch
- Pyhton3
- Numpy

## Dataset
Used a movie reviews dataset accompanied by labels: **_Positive_** and **_Negative_**

> bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that bromwell high is far fetched . what a pity that it

## Model Performance
> Test loss: 0.523
<br/>Test accuracy: 0.806

## Inference on a test review
**Review**: "The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow."


**Model Prediction**:
> Prediction value, pre-rounding: 0.005843
<br/>Negative review detected.

## Acknowledgements
- [Udacity Deep Learning](https://github.com/udacity/deep-learning-v2-pytorch/)
