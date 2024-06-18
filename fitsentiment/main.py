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
    print(logits)
    predicted_class = ''
    predicted_label = np.argmax(logits) # assume label is 1
    for key, value in WORKOUT_CLASSES_VOCAB.items():
        if value == predicted_label:
            predicted_class = key
    return predicted_class

def predict(model, sentence):
    # defining the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    
    # create a new text pipeline
    pipeline = TextPipeline()
    
    # converting the sentence into a df for easier processing
    data = {'text': [sentence]}
    df = pd.DataFrame(data)
    
    # getting the processed and encoded sentence
    encoded_sentence = pipeline.process_data(df=df)
    print('encoded sentence: ', encoded_sentence)
    length_encoded_sentence = [len(encoded_sentence)]
    
    print('length: ', length_encoded_sentence)
    
    # reshaping tensor and getting the length
    tensor = torch.LongTensor(encoded_sentence).to(device)
    
    print('old tensor ', tensor)
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch, number of words
    
    print('updated tensor: ', tensor)
    tensor_length = torch.LongTensor(length_encoded_sentence)      
    
    print('tensor length: ', tensor_length)           
    
    # making the prediction and getting the corresponding class
    logits = model(tensor, tensor_length)
    prediction = get_predicted_class(logits.cpu().detach().numpy())
    return prediction   

if __name__ == '__main__':
    sentence = 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs'
    path_to_model = 'model/saved_model.pt'
    model = load_model(path_to_model)
    prediction = predict(model, sentence)
    print(f'Sentence: {sentence} \nPrediction: {prediction}')