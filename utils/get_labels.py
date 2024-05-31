"""_summary_


Source: to help label the data to be positive, netural or negative: https://medium.com/analytics-vidhya/sentiment-analysis-with-vader-label-the-unlabeled-data-8dd785225166
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from constants.test_constants import TEST_CORPUS

def _get_vader_sentiment(sentence):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(sentence)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif (scores['compound'] > -0.05) and (scores['compound'] < 0.05):
        return 'netural'
    else: # scores['compound'] <= -0.05
        return 'negative'

def define_labels(raw_corpus):
    labels = []
    for sentence in raw_corpus:
        sentiment = _get_vader_sentiment(sentence)
        labels.append(sentiment)
    return labels
            
# Usage
labels = define_labels(TEST_CORPUS)
print('label length: ', len(labels))
print('corpus length: ', len(TEST_CORPUS))
print()

print(labels)

print(list(zip(TEST_CORPUS, labels)))
