import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import nltk
from nltk.stem.snowball import SnowballStemmer
from spacy.lemmatizer import Lemmatizer
import time
import numpy as np


# a dummy function that just returns its input
def identity(x):
    return x


def word_to_sentence(doc):
    return ''.join(doc)


# pos tagger as feature
def tokenize_pos(tokens):

    nltk.download('averaged_perceptron_tagger')

    return [token + "_POS-" + tag for token, tag in nltk.pos_tag(tokens)]


#spacy lemmatization for lemmatization feature
class SpacyLemmatizer(object):
    def __init__(self):
        self.spcyL = Lemmatizer()

    def __call__(self, doc):
        return [self.spcyL.lemmatizer(t) for t in doc]


# stemmer for word ngrams using SnowBall Stemmer
def apply_word_stemmer(doc):

    stemmer = SnowballStemmer(language='english')

    return [stemmer.stem(word) for word in doc]



# stemmer for character ngrams using SnowBall Stemmer
def apply_char_stemmer(doc):

    stemmer = SnowballStemmer(language='english')

    return ''.join([stemmer.stem(word) for word in doc])


# based on the value of tfidf (True/False), Select TF-IDF or Count Vectorizer
def tf_idf_func(tfidf):
    # a dummy function as tokenizer and preprocessor, since the texts are already preprocessed and tokenized
    # All type of feature set mentioned in Report tested here
    if tfidf:

        # vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

        # vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity, ngram_range = (1, 3))
        #  vec = TfidfVectorizer(preprocessor = apply_word_stemmer, tokenizer = identity, ngram_range=(1, 3))
        # vec = TfidfVectorizer(analyzer=identity, preprocessor=identity, tokenizer=SpacyLemmatizer, ngram_range=(1, 3))
        # vec = TfidfVectorizer(preprocessor=tokenize_pos, tokenizer=identity)
        #below is the best one for our classifier which is character ngram
        vec = TfidfVectorizer(preprocessor = word_to_sentence,
                              tokenizer = identity,
                              analyzer='char', ngram_range = (1, 7))

        # vec = TfidfVectorizer(preprocessor = apply_char_stemmer,
        #                       tokenizer = identity,
        #                       analyzer='char', ngram_range = (1, 7))
    else:
        # vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
        # vec = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(1, 3))
        # vec = CountVectorizer(preprocessor=apply_word_stemmer, tokenizer=identity, ngram_range=(1, 3))
        # vec = CountVectorizer(analyzer= identity, preprocessor=identity, tokenizer=SpacyLemmatizer, ngram_range=(1, 3))
        # vec = CountVectorizer(preprocessor=tokenize_pos, tokenizer=identity)
        # vec = CountVectorizer(preprocessor = word_to_sentence,
        #                       tokenizer = identity,
        #                       analyzer='char', ngram_range = (1, 7))
        # vec = CountVectorizer(preprocessor=apply_char_stemmer,
        #                       tokenizer=identity,
        #                       analyzer='char', ngram_range=(1, 7))
        vec = CountVectorizer(preprocessor=tokenize_pos, tokenizer=identity)

    return vec


# Using a SVM Linear Kernel
# SVM Classifier: the value of boolean arg - tfIdf (True/False)
# This fuction uses the Linear SVM to train returns a model
def SVM_Linear(trainDoc, trainClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf)

    # combine the vectorizer with the classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.SVC(kernel='linear', C=2.6))] )

    t0 = time.time()
    # Fit/Train classifier according to trainDoc, trainClass
    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    train_time = time.time() - t0
    print("\nTraining Time: ", train_time)

    return classifier


# SVM classifiers results for different values of C
# This function finds the best C value
def SVM_loop(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)
    vec = tf_idf_func(tfIdf)

    # change the range accordingly(start, stop, step_size)
    c_values = list(np.arange(.1, 15, 0.1))
    accu_scores = []
    f1_macro = []

    title = 'Binary(OFF/NOT) + Linear SVM + {0}'.format("TfidfVectorizer" if(tfIdf) else "CountVectorizer")
    print("\n##### Output of {} \n For different values of C ({}-{}) #####".format(title, c_values[0], c_values[-1]))

    for c in c_values:

        # combine the vectorizer with the classifier
        classifier = Pipeline( [('vec', vec),
                                ('cls', svm.SVC(kernel='linear', C=c))] )

        classifier.fit(trainDoc, trainClass)
        testGuess = classifier.predict(testDoc)

        # accumulate all the values of accuracy and f1 so that we can draw the plot later
        accu_scores.append(accuracy_score(testClass, testGuess))
        f1_macro.append(f1_score(testClass, testGuess, average='macro'))

        print("C=", round(c,1),"   Accuracy=", accu_scores[-1],"     F1(macro)=", f1_macro[-1])

    Eval_Matrics.draw_plots(c_values, accu_scores, f1_macro, value_name='C')

#This function runs the classifier on the development data
def classify_dev_set(classifier, devDoc, devClass, title):

    t1 = time.time()
    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(devDoc)

    print("\n########### {} ###########".format(title))

    # This function visualizes the output
    Eval_Matrics.calculate_measures(classifier, devClass, testGuess, title)

    test_time = time.time() - t1
    print("\nTesting Time: ", test_time)

#This function Test the classifier/model using the test data.
def classify_test_set(classifier, testDoc):

    t1 = time.time()

    # Save output class labels in testGuess
    testGuess = classifier.predict(testDoc)

    print("\n\n Test Started...")

    test_time = time.time() - t1
    print('\nTest Done!!')
    print("\nTesting Time: ", test_time)

    return testGuess


# This function runs the Model with Tf-Idf/Count Vectorizer
def run_model(trainDoc, trainClass):

    # Try different values of C to find out best C value
    # SVM_loop(trainDoc, trainClass, testDoc, testClass, tfIdf=False)

    print('\n\n Training Started...')
    classifier = SVM_Linear(trainDoc, trainClass, tfIdf=True)

    return classifier
