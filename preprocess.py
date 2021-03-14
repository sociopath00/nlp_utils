import re
import string
import pickle

import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from config import *

# download nltk dependencies
nltk.download('popular', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# load sentiment analyzer
sia = SentimentIntensityAnalyzer()

# load the model
model = pickle.load(open(MODEL, 'rb'))
vectorized_model = pickle.load(open(VECTORIZED_MODEL, 'rb'))


def preprocess(text):
    """
    "Objective: Text Preprocessing using NLP
    :param text: Text(str) to be preprocess
    :return: processed text
    """
    # load stopwords
    stop_words = set(stopwords.words('english'))

    # remove emojis from text
    regex_pattern = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002500-\U00002BEF"  # chinese char
                                       u"\U00002702-\U000027B0"
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001f926-\U0001f937"
                                       u"\U00010000-\U0010ffff"
                                       u"\u2640-\u2642" 
                                       u"\u2600-\u2B55"
                                       u"\u200d"
                                       u"\u23cf"
                                       u"\u23e9"
                                       u"\u231a"
                                       u"\ufe0f"  # dingbats
                                       u"\u3030"
                                       "]+", flags=re.UNICODE)
    text = regex_pattern.sub(r'', text)

    # convert text to lowercase
    text = text.lower()

    # replace contractions
    for key, value in CONTRACTIONS.items():
        text = text.replace(key, value)

    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize the text
    tokens = word_tokenize(text)

    # remove stop words and lemmatize the text
    cleaned_text = []
    for token, tag in pos_tag(tokens):
        # ignore if token in stopwords and it's not equal to `not`
        if token in stop_words and token not in ('not', 'nor'):
            continue

        # find out pos tag for tokens
        if token.endswith('ing') or tag.startswith('VB'):
            pos = 'v'
        elif tag.startswith("NN"):
            pos = 'n'
        else:
            pos = 'a'

        # lemmatize the text and append it to clean text
        lemma = WordNetLemmatizer()
        token = lemma.lemmatize(token, pos)

        cleaned_text.append(token)

    return " ".join(cleaned_text)


def sentiment(text):
    """
    :Objective: Find sentiment behind the text using NLTK sentiment analyzer
    :param text: Review text(str)
    :return: tuple of sentiment and polarity score
    """
    score = sia.polarity_scores(text)["compound"]
    if any(word for word in NEG_WORDS if word in text):
        return 'negative', -0.5
    elif score > 0.05:
        return 'positive', score

    return 'negative', score


def sentiment_svm(text, probability=False):
    """
    :Objective: Find sentiment behind text using trained model(SVM)
    :param text: Review text(str)
    :param probability: return Probability if True
    :return:
    """
    processed_features = vectorized_model.transform(text).toarray()

    y_predictions = model.predict(processed_features)

    if probability:
        return np.round(np.max(model.predict_proba(processed_features), 1), 2)
    return y_predictions


if __name__ == '__main__':
    pass
