import numpy as np
import pandas as pd
import torch 
from fitsentiment.preprocess_data import TextPipeline
from constants.constants import WORKOUT_CLASSES_VOCAB

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

def get_predicted_class(logits):
    predicted_class = ''
    predicted_label = np.argmax(logits) # assume label is 1
    for key, value in WORKOUT_CLASSES_VOCAB.items():
        if value == predicted_label:
            predicted_class = key
    return predicted_class

def predict(model, sentence):
    # defining the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # create a new text pipeline
    pipeline = TextPipeline()
    
    # converting the sentence into a df for easier processing
    data = {'text': [sentence]}
    df = pd.DataFrame(data)
    
    # getting the processed and encoded sentence
    encoded_sentence = pipeline.process_data(df=df)
    length_encoded_sentence = len(encoded_sentence)
    
    # reshaping tensor and getting the length
    tensor = torch.LongTensor(encoded_sentence).to(device)
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch, number of words
    tensor_length = torch.LongTensor(length_encoded_sentence)                 
    
    # making the prediction and getting the corresponding class
    logits = model(tensor, tensor_length)
    prediction = get_predicted_class(logits.item())
    return prediction   

if __name__ == '__main__':
    pass
    # sentence = 'I enjoy training legs'
    # path_to_model = 'model/saved_model.pt'
    # model = load_model(path_to_model)
    # prediction = predict(model, sentence)
    # print(f'Sentence: {sentence} \n Prediction: {prediction}')