from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(text):
    pattern = r"[^\w\s]"
    text = [''.join([char if char.isalnum() else ' ' for char in word]) for word in text.split()]
    text = ' '.join(text)
    text =  re.sub(pattern," ",text)
     # used word_tokenize function to tokenize the text, gives list
 
    tokenized_text = word_tokenize(text.lower())
    # get the stop words
    stop_words = set(stopwords.words('english'))
    # removed stop words
    tokenized_text = [word for word in tokenized_text if word not in stop_words]
    # applying stemming 
    # stemmer = PorterStemmer()
    # tokenized_text = [stemmer.stem(word) for word in tokenized_text]
    # applying lemmatization
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]

    preprocessed_text = ' '.join(tokenized_text)
    
    return preprocessed_text

def preprocessSequential(text, tokenizer, max_length):
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    text_cleaned = text_cleaned.lower().strip()
    sequences = tokenizer.texts_to_sequences([text_cleaned])
    padded_seq = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_seq[0]  # Return the padded sequence