# Malayalam News Classification #
The model predicts the malayalam news articles into the following categories:

* Sports
* Automobile
* Technology
* Entertainment
* Weather
* Health
* Politics
* Business

Malayalam news sites like [Dailyhunt](https://m.dailyhunt.in/news/india/malayalam) and [Southlive](https://www.southlive.in/) were scrapped to fetch the training data. Over 10,000+ malayalam articles across different categories
 were used to train the model and around 950 articles were used to validate the model.

A set of malayalam stop words were manually listed and removed from the training data while pre processing.

Following features were taken into account and fed into various algorithms like Naive Bayes classifier, Support Vector Machine and Convolution Neural Network:

* count vector
* word level tf-idf
* n-gram level tf-idf
* characters level tf-idf
* word embeddings 
* character count
* word count
* word density


Accuracy of **94.19%** and **93.24%** were achieved using count vector and n-gram features respectively over Naive 
Bayes 
classifier.