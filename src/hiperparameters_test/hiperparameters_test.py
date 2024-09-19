### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from src.utils import *
from src.preprocess import build_vocab
from src.metrics import *
import numpy as np
    
def print_topics(ldaModel):
    pprint(ldaModel.print_topics())
        
if __name__ == '__main__':
    num_topics = 3
    texts = load_file(Path.dataset_path)
    tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    print('Builded vocabulary ✅')
    
    left_topics, right_topics = 3, 5
    
    _alpha = list(np.arange(0.1, 1, 0.3))
    _alpha.append('symmetric')
    _alpha.append('asymmetric')

    _beta = list(np.arange(0.1, 1, 0.3))
    _beta.append('symmetric')
    
    coherences = []
    for k in range(left_topics, right_topics + 1):
        for alpha in _alpha:
            for beta in _beta:
                ldaModel = lda(trans_TFIDF, num_topics =num_topics, id2word =_dict, alpha= alpha, beta= beta)
                print(f'Trained model ✅. Alpha: {alpha}')
                
                cohe = coherence(ldaModel, tokenized_texts, _dict)
                result = {
                    'k': k,
                    'alpha': alpha,
                    'beta': beta,
                    'coherence': cohe,
                    'topic_words_freq': [dict(ldaModel.show_topic(topic, topn=50)) for topic in range(0, k)]
                }
                coherences.append(result)
                
    best_model = sorted(coherence, key= lambda _dict: _dict['coherence'], reverse= True)[0]
    
    print(best_model)
        
    topic_word_cloud(None, best_model['k'], best_model['topic_words_freq'])