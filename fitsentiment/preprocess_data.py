
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_data(corpus: list[str]):
    # change the text corpus into a pandas DataFrame
    df = pd.DataFrame(data=corpus, columns=['text'], dtype='string')
    text = df['text']
    
    print()
    print('================= before preprocessing =================')
    print(text)
    print()
    
    # convert the text in the corpus to lowercases
    text = text.str.lower()

    # removing punctuation from the corpus 
    text = text.replace(r'[.,;:!\?"\'`]', '', regex=True)
    # removing special characters from the corpus
    text = text.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
    
     # changing emojis to their text form
    text = text.apply(lambda x: emoji.demojize(x))
    # removing : from the emojis
    text = text.replace(r':', '', regex=True)
    
    # removing links from the corpus
    text = text.replace(r'http\S+|www\.\S+', '', regex=True)
    # removing email addresses from the corpus
    text = text.replace(r'\w+@\w+\.com', '', regex=True)
    
    # removing stop words from the corpus
    stop_words = stopwords.words('english')
    text = text.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in stop_words))
    # applying lemmatization
    word_net_lem = WordNetLemmatizer()
    text = text.apply(lambda sentence: ' '.join(word_net_lem.lemmatize(word, 'v') for word in sentence.split()))
    
    return text    

# Usage example
# print(df['text'].str.lower())
#print(type(df['text']))
TEST_CORPUS = ('😃 I wan\'t to use this so "bad" but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
text = preprocess_data(TEST_CORPUS)
print('================= after preprocessing =================')

print(text[0])
print()

