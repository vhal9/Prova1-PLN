import nltk
import unidecode
import string
import os
from nltk.corpus import stopwords
#necessario para os calculos de frequencia
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('portuguese')

class Preprocessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()
    # remover acentos
    def remove_accents(self, text):
        return unidecode.unidecode(text)
    # remover pontuacoes
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('','',string.punctuation))
    #dividir o texto em sentencas
    def tokenize_sentences(self, text):
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences
    # tokenizar o texto
    def tokenize_words(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens
    #stemizar
    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]
    #colocar as palavras em letras minusculas
    def lowercase(self, text):
        return text.lower()
    #remover stopwords com base na lista de stopwords do nltk
    def remove_stopwords(self, token):
        for word in token:
            if word in stop_words:
                token.remove(word)
        return token
    #calculo de frequencias
    def words_frequency(self, token_corpus):
        vec = CountVectorizer().fit(token_corpus)

        #Here we get a Bag of Word model that has cleaned the text, removing non-aphanumeric characters and stop words.
        bag_of_words = vec.transform(token_corpus)

        #sum_words is a vector that contains the sum of each word occurrence in all texts in the corpus. 
        #In other words, we are adding the elements for each column of bag_of_words matrix.
        sum_words = bag_of_words.sum(axis=0)

        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        most_freq_words = sorted(words_freq, key = lambda x: x[1], reverse=True)
        less_freq_words = sorted(words_freq, key = lambda x: x[1])
        
        return words_freq, most_freq_words[:20], less_freq_words[:20]