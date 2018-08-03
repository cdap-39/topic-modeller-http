import re
import pandas as pd
import json


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords

if __name__ == '__main__':
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Import Data set
    # df = pd.read_json('https://raw.githubusercontent.com/cdap-39/data/master/data.json', orient='records')
    df = pd.read_json('data.json')

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # Simple pre-process
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams(two adjacent words), trigrams(three adjacent words) and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # save dictionary
    id2word.save('id2')

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    import os
    os.environ.update({'MALLET_HOME': r'C:\\mallet-2.0.8\\mallet-2.0.8\\'})

    mallet_path = 'C:\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet'
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)


    # Select the model and print the topics
    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
    model_topics = optimal_model.show_topics(formatted=False)

    # save model to disk
    optimal_model.save('ldamodel')

    from gensim import similarities

    index = similarities.MatrixSimilarity(optimal_model[corpus])
    index.save('index')

    query = "Police say that they have already identified some of the suspects who were involved in killing a leopard in Ambalkulam in Kilinochchi. The suspects have been identified by examining video footage. A senior officer at the Kilinochchi Police stated that investigations are underway to apprehend four such identified suspects. Kilinochchi Magistrate Court yesterday ordered police to examine video footage and arrest the suspects who were involved in clubbing the leopard to death."
    vec_bow = id2word.doc2bow(query.lower().split())

    vec_lda = optimal_model[vec_bow]
    sims = index[vec_lda]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims)

    # print(df['data'][19])
