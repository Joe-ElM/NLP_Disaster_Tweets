# Natural Language Processing (NLP) Exploration Summary: Disaster Tweets

## Introduction

This repository contains a comprehensive exploration of a dataset sourced from the "Natural Language Processing with Disaster Tweets" Kaggle competition. The primary objective is to leverage various Natural Language Processing (NLP) techniques to develop predictive models capable of discerning whether a tweet pertains to a real disaster or not. By delving into the structure of the dataset and analyzing the target variable, this summary extends to include the utilization of NLP methods such as text preprocessing, feature extraction, and modeling with machine learning algorithms.

## Goals of NLP Exploration

- Text Preprocessing and Cleaning:

  - Application of text preprocessing techniques such as lowercasing, tokenization, and removal of stop words, URLs, and special characters.
  - Cleaning and normalization of text data to ensure consistency and enhance model performance.

- Feature Extraction using Bag-of-Words (BoW):

  - Representation of text data as numerical features using the Bag-of-Words (BoW) technique.
  - Analysis of vocabulary size, feature matrix representation, and identification of common words and phrases in both disaster and non-disaster tweets.

- Modeling with BoW Representation:
  - Implementation of machine learning algorithms such as Support Vector Classifier (SVC), Logistic Regression, and Random Forest using BoW features.
  - Hyperparameter tuning, model optimization, and evaluation of model performance metrics such as F1 score, recall, precision, and accuracy.

## Dataset Overview

The dataset comprises tweets labeled as either disaster or non-disaster, along with additional metadata. Key features include:

- `text`: The content of the tweet.
- `target`: Binary label indicating whether the tweet refers to a real disaster (1) or not (0).

## Process

The NLP exploration process encompasses the following steps:

1. Text Preprocessing: The text data undergoes preprocessing to standardize its format and remove noise.
2. Feature Extraction with BoW: The Bag-of-Words (BoW) technique is employed to convert text data into numerical features.
3. Modeling and Evaluation: Various machine learning algorithms are applied to the BoW representation of the text data, and the performance of each model is evaluated using appropriate metrics.
4. Comparison and Selection: The models' performance is compared, and the most effective one is selected for further refinement and deployment.

## Data Preprocessing Functions

### `remove_url(text)`

Function to remove URLs from text.

### `text_clean(text)`

Function to clean and preprocess text data, including removing URLs, stopwords, and punctuation, and lemmatizing tokens.

### `keyword_clean(text)`

Function to clean keyword text by replacing `%20` with spaces.

## Modeling

The following machine learning algorithms are applied to the preprocessed text data:

- Support Vector Classifier (SVC)
- Logistic Regression
- Random Forest

## Visualization

Various visualizations are used to analyze the data and model performance, including:

- Top words for disaster and non-disaster messages

![Top words for disaster messages](/Images/Top_words_for_disaster_messages.png)

![Top words for non-disaster messages](/Images/Top_words_for_No_disaster_messages.png)

## Conclusion

Through this exploration, insights into the structure and content of disaster tweets are gained, facilitating the development of predictive models capable of accurately classifying tweets based on their relevance to real disasters. The utilization of NLP techniques coupled with machine learning algorithms enables the extraction of valuable information from text data, thereby contributing to improved disaster response and management efforts.
