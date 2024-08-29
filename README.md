# FitSentiment

FitSentiment is a sentiment analysis tool designed to analyze discussions about calorie counting apps, identifying the features users care about and understanding the sentiment associated with those features. The classifier will process data from diet-related subreddits, focusing on popular apps like MyFitnessPal, LoseIt! and MacroFactor.

## Background

Extracting features was achieved by creating a custom NER model. The NER Annotator for SpaCy was used to create training data for the custom NER model. The link to the tool is as follows: https://tecoholic.github.io/ner-annotator/

The sentiment anaylsis model is based on an LSTM (long short-term memory) neural network, a type of RNN (recurrent neural network) that can handle long dependencies in sequences and tackles the 
vanishing gradient problem, making it a solid neural network to be used for text classification and sentiment analysis problems. 

The corpus was extracted by creating a scraper to scrape revelant comments from subreddits. The PRAW Reddit API wrapper was used for simple access to the Reddit API.

## Setting up the project

### 1. Create a virtual environment in the directory of the project.
```
python3 -m venv venv
```

### 2. Activate the virtual environment on your operating system.

On macOS/ Linux:

```
source venv/bin/activate
```

On Windows (cmd):

```
.venv\Scripts\activate.bat
```

On Windows (Powershell):

```
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies into your virtual environment.
```
pip install -r requirements.txt
```

### 4. Create an .env file, and paste the following code into the file, replacing the text with your secrets.
``` 
CLIENT_ID=YOUR_CLIENT_ID
CLIENT_SECRET=YOUR_CLIENT_SECRET
USERNAME=YOUR_USERNAME
PASSWORD=YOUR_PASSWORD
```

### 5. Build the project by running the following code in your virtual environment.

```
python setup.py sdist bdist_wheel
```

### 6. Install the project. You can do so by using the name 'FitSentiment', or installing it in editable mode with -e to make changes.

Example: installing the project by its name. 
```
pip install FitSentiment
```

Example: installing the project in editable mode. 
```
pip install -e .
```

## Possible Expansions

1. Visualizing the data by creating a dashboard via Streamlit, Flask or other resources.
2. Adding more training data from other social media platforms, such as Twitter, Meta and YouTube.
