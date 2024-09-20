import spacy 
from utils import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel 

def build_vocab(texts):
    print('Building vocabulary and corpus...')
    
    try:
        nlp = load_file(Path.nlp_path.value)    
        print('Model loaded successfully from local space ✅')
    except:
        nlp = spacy.load("en_core_web_sm")

        save_file(nlp, Path.nlp_path.value)  
        print('Model loaded successfully from remote space ✅')
    
    if texts is None: raise Exception('No text to build vocabulary')
    
    tokenized_texts = [nlp(text) for text in texts]
    tokenized_texts = [[token.lemma_ for token in text if not token.is_stop and not token.is_punct and not token.is_space] for text in tokenized_texts]
    
    # Create a dictionary representation of the documents.
    _dict = Dictionary(tokenized_texts)
    bow_corpus = [_dict.doc2bow(text) for text in tokenized_texts]
    
    TFIDF = TfidfModel(bow_corpus) # Fit TF-IDF model
    trans_TFIDF = TFIDF[bow_corpus] # Apply TF-IDF model
    
    print('Builded vocabulary and corpus ✅.')
    
    return tokenized_texts, _dict, trans_TFIDF