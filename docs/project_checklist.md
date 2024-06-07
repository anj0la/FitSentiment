# FitSentiment Project Checklist

**Anjola Aina**

**FitSentiment**

**27 MAY 2024**

## Overview

FitSentiment is a sentiment analysis classifier designed to analyze discussions about workout splits. It aims to understand public sentiment towards various weekly workout routines. The classifier will be trained using data from fitness-related subreddits such as r/bodyweightfitness, r/Running, r/Fitness, r/bodybuilding, and r/xxfitness.

## The Steps

### 1. Data Collection
Scrape posts and comments from the mentioned subreddits. These posts and comments create the training corpus for the classifier. To find the “best” training split, information will be extracted from the relevant subreddits about people’s weekly workout splits. Keywords such as ‘weekly split’, ‘workout split’, and ‘weekly workouts’ will be used to filter data from these subreddits to only focus on posts and comments related to them.

### 2. Label Data
Label each post or comment with its corresponding sentiment: positive, negative, or neutral.

### 3. Preprocess Data
Preprocess the data by completing the following steps:

1. Convert text to lowercase.
2. Remove punctuation and special characters.
3. Tokenize the text into words.
4. Remove stop words (e.g, ‘and’, ‘the’, ‘is’)
5. Apply lemmatization to reduce words to their base form.

### 4. Building the Classifier
Build an LSTM-based model that learns word embeddings and captures semantic meanings.

### 5. Training and Evaluation
Split the data into training and testing sets (e.g, 80% training, 20% testing), and train the model on the training data. Evaluate the model using the testing data with metrics such as accuracy, precision, recall and F1 score.

---

## The Checklist

| Name                | Description                                                                                     | Completed Date |
|---------------------|-------------------------------------------------------------------------------------------------|----------------|
| **Data Collection** | Scrape posts and comments from the following subreddits: r/Fitness, r/xxfitness, r/Bodybuilding, r/bodyweightfitness, r/Running. These posts and comments should be sentences, so we only care about the ‘text’ field from the Reddit API call. | 05/29/2024     |
| **Label Data**      | Automatically label each post or comment based on predefined keywords.                          | 06/02/2024     |
| **Preprocess Data** | Convert text to lowercase. Remove punctuation and special characters. Remove stop words (e.g., ‘and’, ‘the’, ‘is’). Apply lemmatization to reduce words to their base form. Tokenize the text into words. | 05/30/2024     |
| **Building the Classifier** | Build the LSTM model. Create the initial word embeddings of the input (represented in a numerical format). Create the LSTM layer. Create a feed-forward (MLP) network to add nonlinearities. | 06/05/2024     |
| **Training and Evaluation** | Train the classifier. Evaluate the model using the testing data with metrics such as accuracy, precision, recall and F1 score. Save the best model. |                |

To view the file, use the link: https://docs.google.com/document/d/1dKI85bHRNh7Tcon1mUkkQ7oQgoSkvmit08LQmu_dl1Q/edit?usp=sharing