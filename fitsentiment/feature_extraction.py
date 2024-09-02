"""
File: feature_extraction.py

Author: Anjola Aina
Date Modified: September 2nd, 2024

This file contains all of the necessary functions to build the custom NER model used to extrract features from the text.
These features are: APP and FEATURE, where APP is a calorie counting application and FEATURE are specific features of the app (e.g., macronutrient tracking) 
respectively.

The following source was used as a guide to build the NER model: https://medium.com/@mjghadge9007/building-your-own-custom-named-entity-recognition-ner-model-with-spacy-v3-a-step-by-step-guide-15c7dcb1c416

Functions:
    TODO: Populate this section with functions.
"""
import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

def get_spacy_doc(log_file_path: str, data):
    """
    This function creates spaCy DocBin onjects from annotated data.
    
    It prepares data for training a custom NER model by transforming annotated data
    into a format compatible with spaCy v3, creating Doc objects with character spans linked
    to entities.
    
    Entity overlap is prevented to avoid training issues.
    
    Data issues are logged and captured in the log_file to aid with debugging and data quality assessment.

    Args:
        log_file_path (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # create a blank spaCy pipeline
    nlp = spacy.blank('en')
    db = DocBin()
    
    # iterate through the data
    for text, annot in tqdm(data):
        doc = nlp.make_doc(text) # create doc object from the text
        annot = annot['entities']
        
        # extract entities from the annotations
        ents = extract_entities(log_file_path, text, doc, annot)
        
        # attempt to label the text with the entities and add it to the docbin object
        try:
            doc.ents = ents 
            db.add(doc)
        except:
            # no entities were extracted from the text, don't add this entity to the docbin object
            pass 
        
def extract_entities(log_file_path, text, doc, annot):
    ents = []
    ent_indices = []
    
    # open the log file for writing
    f = open(file=log_file_path, mode='w') 
    
    for start, end, label in annot:
        # check if the current entity overlaps with any previously processed entity
        overlap = any(idx in ent_indices for idx in range(start, end))
        
        # if there's no overlap, process the entity
        if not overlap:
            # add the indices of the current entity to ent_indices to track its span
            ent_indices.extend(range(start, end))
      
        # attempt to create a span for the current entity in the document
        try:
            span = doc.char_span(start, end, label=label, alignment_mode='strict')
        except:
            pass # ignore the problematic entity
        
        if not span:
            # log errors for annotations that couldn't be processed
            err_data = str([start, end]) + '    ' + str(text) + '\n'
            f.write(err_data)
        else:
            ents.append(span)

    # close the log file
    f.close()
    
    return ents