import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer

# Custom modules
import project.server.main.mrc_stopwords as mrc_stopwords

stopwords = mrc_stopwords.stopwords
lem = WordNetLemmatizer()

# UNUSED FUNCTION (FOR NOW)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    # use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a'
    elif nltk_tag.startswith('V'):
        return 'v'
    elif nltk_tag.startswith('N'):
        return 'n'
    elif nltk_tag.startswith('R'):
        return 'r'
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    words = set([w.lower() for w in lemmatized_sentence if any(c.isalpha() for c in w)])
    words = [w for w in words if w not in stopwords]
    return words

def get_keywords_from_question(question):
    """
    Given a question (str), this function returns two sets of strings.
    The first one contains the keywords of the question that have not been
    lemmatized, and the second one contains the keywords that have been
    lemmatized.
    """
    # Only keep letters, numbers and hyphens
    text = re.sub('[^A-Za-z0-9\-\']', ' ', question)
    # Convert to lowercase
    text = text.lower()

    # Extract set of (lemmatized) keyworeds
    text = text.split()
    keywords = set(word for word in text if not word in stopwords)
    lemmatized_keywords = set()
    words_to_remove = set()
    for word in keywords:
        lemm_word = lem.lemmatize(word)
        if not lemm_word in stopwords:
            lemmatized_keywords.add(lemm_word)
        if lemm_word != word:
            words_to_remove.add(word)
    keywords = set(word for word in keywords if not word in words_to_remove)
    return keywords, lemmatized_keywords

def extract_keywords(question, topn=10):
    """
    Given a question (str), this function returns a list of the keywords present
    in the question.
    """
    keywords, lemmatized_keywords = get_keywords_from_question(question)

    # WE PLAN TO SCORE EACH KEYWORD AND RETURN THEM IN AN ORDERED FASHION. TODO

    #text = ' '.join(text)
    #keywords = extract_topn_from_vector(feature_names, sorted_items, topn)

    return list(keywords), list(lemmatized_keywords) # Set union operator
