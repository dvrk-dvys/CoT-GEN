from gensim import corpora
from gensim.models import Word2Vec, LdaModel

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
#nltk.download('vader_lexicon')
import spacy
nltk.download('punkt_tab')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_lg")





def analyze_sentiment(text, aspects):
    for aspect in aspects:
        if aspect in text:
            sentence = next(sent for sent in nltk.sent_tokenize(text) if aspect in sent)
            sentiment = sia.polarity_scores(sentence)['compound']
            print(f'Sentiment for {aspect}: {sentiment}')


def extract_aspect_features(text):
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'attr', 'nmod'] and token.pos_ == 'NOUN':
            aspects.append(token.text)
    return aspects


def analyze_implicit_sentiment(text, aspect):
    doc = nlp(text)
    for token in doc:
        if token.text.lower() == aspect.lower():
            for child in token.children:
                if child.pos_ == "ADJ":
                    return child.text

    return 'False'


def extract_aspect_opinion_pairs(text):
    doc = nlp(text)
    pairs = []
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"] and token.pos_ == "NOUN":
            for child in token.children:
                if child.pos_ == "ADJ":
                    pairs.append((token.text, child.text))
    return pairs


def wv_context_understanding(sentences, target):
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    #print("Word Vector Context", model.wv.most_similar(target))
    return model.wv.most_similar(target)

def analyze_contextual_sentiment(text, aspect, window_size=5):
    words = nltk.word_tokenize(text)
    aspect_index = words.index(aspect)
    start = max(0, aspect_index - window_size)
    end = min(len(words), aspect_index + window_size + 1)
    context = ' '.join(words[start:end])

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(context)['compound']
    return sentiment

def handle_negations_intensifiers(text):
    words = nltk.word_tokenize(text)
    negations = {"not", "no", "never", "neither", "hardly", "scarcely"}
    intensifiers = {"very", "extremely", "incredibly", "absolutely"}

    modified_text = []
    negate = False
    intensify = False

    for word in words:
        if word.lower() in negations:
            negate = True
        elif word.lower() in intensifiers:
            intensify = True
        else:
            if negate:
                word = "NOT_" + word
                negate = False
            if intensify:
                word = "INTENSIFIED_" + word
                intensify = False
        modified_text.append(word)
    return " ".join(modified_text)


def detect_sarcasm(text):
    sia = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)

    conflicting_sentiments = False
    overall_sentiment = sia.polarity_scores(text)['compound']

    for sentence in sentences:
        sentence_sentiment = sia.polarity_scores(sentence)['compound']
        if (overall_sentiment > 0 and sentence_sentiment < -0.5) or (overall_sentiment < 0 and sentence_sentiment > 0.5):
            conflicting_sentiments = True
            break

    if conflicting_sentiments:
        return "Potential Sarcasm detected"
    else:
        return "No sarcasm detected"

#Latent Dirichlet Allocation
def extract_aspects_lda(documents, num_topics=5, num_words=5):
    stop_words = set(stopwords.words('english'))
    texts = [[word for word in nltk.word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words] for doc in documents]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    topics = lda_model.print_topics(num_words=num_words)


    #for topic in topics:
        #print(topic)
    return topics


if __name__ == '__main__':
    text = "The phone's camera is excellent, but the battery life is disappointing."
    sia = SentimentIntensityAnalyzer()
    aspects = ['camera', 'battery']
    analyze_sentiment(text, aspects)
    print('----------------')


    text_1 = "The restaurant's ambiance was cozy, and the food was delicious."
    print(f"aspect features: {extract_aspect_features(text_1)}")
    print('----------------')

#    <sentence id="958:1">
#        <text>Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag.</text>
#        <aspectTerms>
#            <aspectTerm term="internal speakers" polarity="negative" from="86" to="103" implicit_sentiment="False" opinion_words="lousy"/>
#            <aspectTerm term="price tag" polarity="positive" from="203" to="212" implicit_sentiment="True"/>
#            <aspectTerm term="click pads" polarity="negative" from="30" to="40" implicit_sentiment="True"/>
#        </aspectTerms>
#    </sentence>


    text_2 = "The phone's camera captures stunning details."
    text_3 ="Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag."
    print("implicit sentiment: ", analyze_implicit_sentiment(text_2, "camera"))
    print("implicit sentiment: ", analyze_implicit_sentiment(text_3, "click pads"))
    print("implicit sentiment: ", analyze_implicit_sentiment(text_3, "price tag"))
    print('----------------')

#!!!!TOKENIZER DOESNT HANDLE MULTWORD TOKENS
    sentences = ["The camera quality is amazing",
                 "The battery life is disappointing",
                 "The screen resolution is impressive"]

    print("Word Vector context: ", wv_context_understanding(sentences, "camera"))
    print('----------------')

    text_4 = 'The phone has a bright screen and a powerful processor.'
    print("Aspect Opinion Pairs: ", extract_aspect_opinion_pairs(text_4))
    print('----------------')


    text_5 = "Despite its small size, tthe camera produces excellent quality photos."
    print('contextual sentiment: ', analyze_contextual_sentiment(text_5, 'camera'))
    print('----------------')



    text_6 = "The camera is not very good, but the screen is extremely bright."
    modified_text = handle_negations_intensifiers(text_6)
    print("Handle Negations and Intensifiers: ", modified_text)
    print('----------------')


    #sia = SentimentIntensityAnalyzer()
    print("Polarity scores: ", sia.polarity_scores(modified_text))
    print('----------------')


    text_7 = "The weather is just perfect today. I just love getting soaked in the rain."
    text_8 = "The new feature is so useful. It only crashed my computer twice today."

    print("Detect Sarcasm", detect_sarcasm(text_7))
    print("Detect Sarcasm", detect_sarcasm(text_8))
    print('-------broken!---------')

    #BROKEN!


    documents = [
        "The camera on this phone is amazing. Great picture quality.",
        "Battery life could be better. It drains quickly.",
        "The screen is bright and crisp. Colors look fantastic.",
        "Processing speed is impressive. Apps load quickly",
        "The design is sleep and modern. Feels great in hand."
    ]

    print("Extract Latent Dirichlet Allocation Aspects: ", extract_aspects_lda(documents))



