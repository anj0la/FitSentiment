# FitSentiment Project Checklist

**Date Created: May 27, 2024**

**Last Updated: August 28, 2024**

## Overview

FitSentiment is a text classifier designed to analyze discussions about workout splits. It aims to understand public sentiment towards various weekly workout routines. The classifier is trained using data from fitness-related subreddits such as r/bodyweightfitness, r/Running, r/Fitness, r/bodybuilding, and r/xxfitness.
FitSentiment is a sentiment analysis tool designed to analyze discussions about calorie counting apps, identifying the features users care about and understanding the sentiment associated with those features. The classifier will process data from diet-related subreddits such as r/cico, r/caloriecount, r/loseit, and r/1200isplenty.

## The Steps

### 1. Data Collection
Scrape posts and comments from specific subreddits. These posts and comments create the training corpus for the project. Only comments pertaining to popular apps such as 'MyFitnessPal' and 'Loseit!' will be collected.

### 2. Preprocess Data 
Preprocess the scraped data by completing the following steps:

1. Convert text to lowercase.
2. Remove punctuation and special characters.
3. Remove emails and links.
4. Remove emojis and unicode characters.
5. Remove stop words (e.g., ‘and’, ‘the’, ‘is’).
6. Tokenize the text into words.
7. Build the vocabulary.

### 3. Feature Extraction
Complete the following steps to extract features and create training data:

1. Identify and extract specific features mentioned in relation to each app (e.g., macronutrient tracking, barcode scanner) using 
a custom NER (named entity recognition) model. 
2. Create the feature set (app, context, feature), where the context refers to the words surrounding the feature.

At this point, the data has been converted into the format used as the input to the sentiment analysis model. It still needs to be encoded before it can be used as an input to the model, but the input consists of three items (app, context, feature) with the shape (3,).

Note that if no feature exists for a given sentence, the feature is set to None, and the context is the full sentence.
If more than one feature is mentioned in a given sentence, the context will end before the next feature is mentioned.

### 4. Building the Model
Build an LSTM-based model that learns word embeddings and captures semantic meanings.
The input to the model is of shape (3,), consisting of the app, the context and the feature.

Note that if the feature is None, sentiment analysis will be done on the full sentence.

### 5. Training and Evaluation
Split the data into training and testing sets (e.g, 80% training, 20% testing), and train the LSTM model to output a sentiment score (positive, negative) for each feature within its specific context.

Evaluate the model using the testing data with metrics such as accuracy, precision, recall and F1 score.

---

## The Checklist

| Name                | Description                                                                                     | Completed Date |
|---------------------|-------------------------------------------------------------------------------------------------|----------------|
| **Data Collection** | Scrape posts and comments from the following subreddits: r/cico, r/caloriecount, 'loseit', '1200isplenty', 'intermittentfasting', and 'nutrition'. These posts and comments should be sentences, so we only care about the ‘text’ field from the Reddit API call. | 08/26/2024     |
| **Preprocess Data** | Convert text to lowercase. Remove punctuation and special characters. Remove emails and links. Remove emojis and unicode characters. Remove stop words (e.g., ‘and’, ‘the’, ‘is’).                        | 08/27/2024     |
| **Feature Extraction** | Identify and extract specific features mentioned in relation to each app using a custom NER model. Create the feature set. |     |
| **Building the Model** | Build the LSTM model. Create the initial word embeddings of the input (represented in a numerical format). Create the LSTM layer. Create a feed-forward (MLP) network to add nonlinearities. |     |
| **Training and Evaluation** | Train the model. Evaluate the model using the testing data with metrics such as accuracy, precision, recall and F1 score. Save the best model. |                |
