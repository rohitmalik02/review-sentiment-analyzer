# review-sentiment-analyzer
* Flask API to perform Sentiment Analysis on a given user review of a restaurant

* Used Python NLTK and Contractions module to preprocess text and sklearn module to implement a Bag of Words model (vectorization).

* Used Support Vector Machine Classifier with Radial Bias Function kernel to predict the sentiment.

* Performance on validation data set (Accuracy):
    * SVM: 0.78  
    * Gaussian Naive Bayes: 0.73
    * Random Forest: 0.72
