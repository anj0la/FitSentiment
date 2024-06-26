# FitSentiment

FitSentiment is a project designed to analyze discussions about workout splits, determining which workout split has the most positive sentiments. This is done by using a custom text classifier to classify the text into 6 classes (lower body, upper body, full body, upper lower, push pull legs, and general fitness), and passing the classified text into a sentiment analyzer and counting the number of positive and negative sentiments for each class.

## Background

The model is based on an LSTM (long short-term memory) neural network, a type of RNN (recurrent neural network) that can handle long dependencies in sequences and tackles the 
vanishing gradient problem, making it a neural network to be used for text classification problems. 

The corpus was extracted by creating a scraper to scrape comments from subreddits on Reddit that were related to workout splits and weekly routines. The PRAW Reddit API wrapper
was used for simple access to the Reddit API.

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

1. Adding more training data from other social media platforms, such as Twitter, Meta and YouTube.
2. Collecting training data from various onlne bodybuilding / fitness forums.
3. Creating a custom sentiment analysis classifier.