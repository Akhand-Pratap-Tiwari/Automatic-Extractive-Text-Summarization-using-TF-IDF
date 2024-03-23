#Text Summarizer | Type = Extractive
#Using Term Frequency - Inverse Document Frequency Technique

# from nltk import warnings
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print('punkt')
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print('stopwords')
    nltk.download('stopwords')


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def remove_punctuation_marks(text):
    punctuation_marks = dict((ord(punctuation_mark), None)
                             for punctuation_mark in string.punctuation)
    return text.translate(punctuation_marks)


def get_lemmatized_tokens(text):
    normalized_tokens = nltk.word_tokenize(
        remove_punctuation_marks(text.lower()))
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]


def get_average(values):
    greater_than_zero_count = total = 0
    for value in values:
        if value != 0:
            greater_than_zero_count += 1
            total += value
    return total / greater_than_zero_count


def get_threshold(tfidf_results):
    i = total = 0
    while i < (tfidf_results.shape[0]):
        total += get_average(tfidf_results[i, :].toarray()[0])
        i += 1
    return total / tfidf_results.shape[0]


def get_summary(documents, tfidf_results, handicap=0.85):
    summary = ""
    i = 0
    while i < (tfidf_results.shape[0]):
        if (get_average(tfidf_results[i, :].toarray()[0])) >= get_threshold(tfidf_results) * handicap:
            summary += ' ' + documents[i]
        i += 1
    return summary

# Use below function for calling from different file | Pass Text and handicap and then get Summary
def summary_func(text="", handicap=0.9):  
    documents = nltk.sent_tokenize(text)

    tfidf_results = TfidfVectorizer(tokenizer=get_lemmatized_tokens, stop_words=stopwords.words(
        'english')).fit_transform(documents)
    return get_summary(documents, tfidf_results, handicap)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")

    text = '''This product is good for students for Study You can not play heavy gamesike GTA 5 on this laptop.
              Nice product. good for students but can improve in screen resolution.
              Quite dissatisfied with the screen quality overall it's all fine.
              Nothing great to talk about! The specs are mediocre and the screen quality is passable.
              The sound has a lot more to improve.
              The size is quite ok for a satchel/ backpack and is lightweight!
              But I believe there are better offers.
              Good for light personal use only like browsing and office work!
              ASUS VIVOBOOK15 (2021)CELERON N4020, is not adequate for our office purpose we want to return it or exchange it for one with i5 or i3 processor.
              need to know where to send this bought item for refund or exchange.
              Expecting a response at the earliest. Just Go For It Dont Think. The best laptop for students.'''

    documents = nltk.sent_tokenize(text)
    stop_words_list = stopwords.words('english').extend([get_lemmatized_tokens(i) for i in stopwords.words('english')]) 
    tfidf_results = TfidfVectorizer(tokenizer=get_lemmatized_tokens, stop_words=stopwords.words(stop_words_list)).fit_transform(documents)
    print(get_summary(documents, tfidf_results, 0.9))   #Handicap Parameter = 0.9
                                                        #Length of Summarized text is inversely prop. to Handicap  
