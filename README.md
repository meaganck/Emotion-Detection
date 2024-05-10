# Reproduction of Detecting Emotions in Text with GloVe Embeddings
Meagan Choo-Kang
## Background
This project centers on the GloVe model for language representation, which was
developed by Pennington, Socher, and Manning (2014). The main concept of GloVe is rooted in
the idea that the ratios of word-to-word co-occurrence probabilities have the potential to carry
meaning. In GloVe, the training objective is to learn word vectors where the dot product reflects
the logarithm of the probability of word co-occurrences. This approach links vector differences
in the word space with the logarithm of co-occurrence probability ratios. Since these ratios can
convey meaning, this information is also encoded as vector differences.  
\
This unsupervised model was designed to combine the benefits of two other popular
semantic vector space models in the literature: global matrix factorization and local context
window methods. While global matrix factorization models excel at leveraging statistical
information, they perform poorly on word analogy tasks due to their utilization of a sub-optimal
vector space structure. On the other hand, models utilizing the local context window method
perform better on analogy tasks but struggle with utilizing corpus statistics because they train on
separate local context windows. Thus, GloVe was developed to effectively utilize statistics by
training on global word-word co-occurrence counts and to produce a meaningful vector space
structure. In the study conducted by Pennington, Socher, and Manning (2014), GloVe
outperformed other models on tasks involving word analogy, word similarity, and named entity
recognition.  
\
For this project, I will attempt to reproduce the study by Gupta et al. (2021), which
decoded emotions in text by using GloVe embeddings and passed these through a Long ShortTerm Memory (LSTM) based model. Their model classified six emotions: anger, fear, joy, love,
sadness, and surprise, based on the Contextualized Affect Representations for Emotion
Recognition (CARER) model proposed by Saravia et al. (2018). The study reported an overall F1
score of 0.93. Although the model in this project will not be identical to the original one, I hope
to achieve a similar F1 score. Additionally, their model demonstrated superior performance in
predicting joy and sadness compared to other emotions but faced challenges in detecting
surprise.

## Description
*Task*  
The objective of this project is to replicate the study conducted by Gupta et al. (2021).
Therefore, the aim is to develop an LSTM-based model utilizing GloVe embeddings to identify
the following emotions from tweets: anger, fear, joy, love, sadness, and surprise. By replicating
Gupta et al.'s (2021) study, the outcomes of this project could contribute to enhancing
understanding and support in determining the effectiveness of an LSTM-based model integrated
with GloVe embeddings. This, in turn, may offer valuable insights for future research endeavors
in this field.  
\
*Dataset*  
Although the original paper utilized tweets obtained from the Twitter API and classified
the emotions of the tweets based on hashtags, I was unable to access their dataset. Therefore, I
utilized the dataset from Gupta (2021), which also comprises tweets along with their categorized
emotions. However, Gupta, P. (2021)'s dataset classified emotions differently compared to the
original paper. Specifically, Gupta, P. (2021)'s dataset included 13 emotions: empty, sadness,
enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger. Most
of these emotions were similar to those in the original paper, except for fear and joy. To
substitute for 'fear', I used 'worry', and for 'joy', I used 'happiness' in Gupta, P. (2021)'s dataset.
Additionally, the other emotions in the dataset were omitted. since this project will only focus on
the emotions covered in the original paper.  
\
Furthermore, the tweets were cleaned-up by removing stop words, such as words used for
grammatical structure and pronouns. Since this is a preliminary project, future work should also
remove the Twitter usernames and shortened words that have repeating characters (for example,
“hmmmm”). Additionally, concerning the target data, the emotions were one-hot-encoded to
make it easier for the model to predict the emotions. Also, similar to the original study, the data
was split with 90% for training and 10% for testing, even though they used a different dataset.  
\
*Model*  
The model replicated the approach outlined by Pennington, Socher, and Manning (2014).
Firstly, GloVe was used to create a 2D matrix of vectors, containing each word in every sentence
of the dataset. The specific version of GloVe utilized by Pennington, Socher, and Manning
(2014) was glove.6B.100d, which will also be adopted in this project.
Secondly, they constructed a Long Short-Term Memory Network (LSTM), a type of
Recurrent Neural Network (RNN) chosen for its capability to learn long-term dependencies.
Their LSTM comprised a ReLU activation function, a pooling layer, two hidden layers with 128
nodes each, and an output layer with SoftMax. The model employed the Adam optimizer,
utilized batches of 120 for each iteration during training, and implemented a dropout rate of 0.5.
Since the original paper did not specify the number of iterations employed, this project will
utilize 5 epochs for training. Ideally, the model would undergo training for more epochs;
however, since this is just initial project and it takes long to train, I will just use 5 epochs for
now.  
\
*Evaluation*  
To evaluate the performance of this model, F1-score will be used since it is a common
evaluation metric used in emotion recognition studies because emotion datasets tend to be
unbalanced (Savavia et al., 2018).  
\
Another potential method for evaluating performance is to compare the model's
performance with that of human participants. For instance, both the model and human
participants can be tasked with classifying given sentences into one of the six emotions: anger,
fear, joy, love, sadness, and surprise. If the model's results closely resemble those of human
participants, it can provide further support for the model's potential success in replicating
emotion detection in humans.

## References
Grupta, P. (2021). Emotion Detection from Text. Kaggle.
<https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-fromtext?resource=download>  
\
Gupta, P., Roy, I., Batra, G., Dubey, A.K. (2021). Decoding Emotions in Text Using GloVe
Embeddings. In 2021 International Conference on Computing, Communication, and
Intelligent Systems (ICCCIS), Greater Noida, India, pp. 36-40.
<https://doi.org/10.1109/ICCCIS51004.2021.9397132>  
\
Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for World
Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP), Doha, Qatar, pp. 1532–1543.
<https://doi.org/10.3115/v1/D14-1162>  
\
Savavia, E., Liu, H.T., Huang, Y., Wu, J., Chen, Y. (2018). CARER: Contextualized Affect
Representations for Emotion Recognition. In Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing, Brussels, Belgium, pp. 3687-3697.
<https://doi.org/10.18653/v1/D18-1404>
