### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from src.utils import *
from src.preprocess import build_vocab
from src.metrics import *
import numpy as np
import sys
    
def print_topics(ldaModel):
    pprint(ldaModel.print_topics())
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("running hyperparameters.py")
        sys.exit(1)

    config_path = sys.argv[1]
    configuracion = cargar_configuracion(config_path)
    
    num_topics = configuracion['k_fit_eval']
    
    left_topics = configuracion['lk_hyperparameters']
    right_topics = configuracion['rg_hyperparameters']
    lalpha = configuracion['lalpha_hyperparameters']
    ralpha = configuracion['ralpha_hyperparameters']
    alpha_step = configuracion['alpha_step_hyperparameters']
    lbeta = configuracion['lbeta_hyperparameters']
    rbeta = configuracion['lbeta_hyperparameters']
    beta_step = configuracion['beta_step_hyperparameters']
    
    _alpha = list(np.arange(lalpha, ralpha, alpha_step))
    _alpha.append('symmetric')
    _alpha.append('asymmetric')

    _beta = list(np.arange(lbeta, rbeta, beta_step))
    _beta.append('symmetric')
    
    texts = load_file(Path.dataset_path)
    tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    print('Builded vocabulary ✅')
    
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