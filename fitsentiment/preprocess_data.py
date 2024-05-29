
import re
import pandas as pd
from utils.constants import TEST_CORPUS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess_data(corpus):
    # convert the text in the corpus to lowercases
    corpus = corpus.str.lower()
    # removing punctuation from the corpus 
    corpus = corpus.replace(r'[.,;:!\?"\'`]', '', regex=True)
    # removing special characters from the corpus
    corpus = corpus.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
    # removing links from the corpus
    corpus = corpus.replace(r'http\S+|www\.\S+', '', regex=True)
    # removing email addresses from the corpus
    corpus = corpus.replace(r'\w+@\w+\.com', '', regex=True)
    # removing stop words from the corpus
    stop_words = stopwords.words('english')
    corpus = corpus.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in stop_words))

    # applying lemmatization
    word_net_lem = WordNetLemmatizer()
    corpus = corpus.apply(lambda sentence: ' '.join(word_net_lem.lemmatize(word, 'v') for word in sentence.split()))
    
    return corpus    

# Usage example
df = pd.DataFrame(data=TEST_CORPUS, columns=['text'], dtype='string')
print('================= before preprocessing =================')
print(df['text'])
print()
# print(df['text'].str.lower())
#print(type(df['text']))
print('================= after preprocessing =================')
new_corpus = preprocess_data(df['text'])
print(new_corpus)
print(new_corpus[5])

