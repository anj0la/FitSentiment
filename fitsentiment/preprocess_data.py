import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text


def preprocess_data(corpus: list[str]):
    # change the text corpus into a pandas DataFrame
    df = pd.DataFrame(data=corpus, columns=['text'], dtype='string')
    data = df['text']
    
    print()
    print('================= before preprocessing =================')
    print(data)
    print()
    
    # convert the text in the corpus to lowercases
    data = data.str.lower()

    # removing punctuation from the corpus 
    data = data.replace(r'[.,;:!\?"\'`]', '', regex=True)
    # removing special characters from the corpus
    data = data.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
    
     # changing emojis to their text form
    data = data.apply(lambda x: emoji.demojize(x))
    # removing : from the emojis
    data = data.replace(r':', '', regex=True)
    
    # removing links from the corpus
    data = data.replace(r'http\S+|www\.\S+', '', regex=True)
    # removing email addresses from the corpus
    data = data.replace(r'\w+@\w+\.com', '', regex=True)
    
    # removing stop words from the corpus
    stop_words = stopwords.words('english')
    data = data.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in stop_words))
    # applying lemmatization
    word_net_lem = WordNetLemmatizer()
    data = data.apply(lambda sentence: lemmatize_text(sentence))
    
    return data.values

def tokenize_data(data) -> tuple[list, dict]:
    vectorizer = TfidfVectorizer()
    # creating the vocabulary by fitting the data
    vectorizer.fit(data)
    vocab = vectorizer.vocabulary_
    
    # tokenizing the data
    tokenized_data = []
    for sentence in data:
        tokenized_data.extend([word_tokenize(term) for term in sent_tokenize(sentence)])
        
    return tokenized_data, vocab

# Usage example
TEST_CORPUS = ('😃 I wan\'t to use this so "bad" but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
text = preprocess_data(TEST_CORPUS)
print('================= after preprocessing =================')

print(text)
print()

print('================= after tokenizing =================')
tokenized_data, vocab = tokenize_data(text)

print('tokenized data: ', tokenized_data)

print()

print('vocabulary: ', vocab)

print()



