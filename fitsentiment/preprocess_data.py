import torch
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

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
    data = data.apply(lambda sentence: ' '.join(word_net_lem.lemmatize(word, 'v') for word in sentence.split()))
    
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

def convert_to_torch(tokenized_data: list, vocab: dict):
    max_length = max(len(sentence) for sentence in tokenized_data)
    num_repr = []
    for sentence in tokenized_data:
        num_sentence = [vocab.get(token, 0) for token in sentence]  # Use 0 for tokens not in vocab
        num_sentence += [0] * (max_length - len(sentence))  # Pad with zeros
        num_repr.append(num_sentence)
    tensor_repr = torch.tensor(num_repr, dtype=torch.long)  # Convert list of lists to tensor
    return tensor_repr

def batchify(tokenized_data, batch_size):
    batches = []
    for i in range(0, len(tokenized_data), batch_size):
        batch = tokenized_data[i:i+batch_size]
        batches.append(batch)
    return batches

# Usage example
# print(df['text'].str.lower())
#print(type(df['text']))
TEST_CORPUS = ('😃 I wan\'t to use this so "bad" but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
text = preprocess_data(TEST_CORPUS)
print('================= after preprocessing =================')

print(text)
print()

print('================= after tokenizing =================')
tokenized_data, vocab = tokenize_data(text)

print('tokenized data: ', tokenized_data)
print('vocabulary: ', vocab)

print()
print('================= after converting to tensor =================')
tensor_input = convert_to_torch(tokenized_data, vocab)
print('tensor input: ', tensor_input)



# Example usage
tokenized_data = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 10], [10, 0, 0, 0, 0]]
batched_data = batchify(tokenized_data, batch_size=1)
for batch in batched_data:
    print(batch)