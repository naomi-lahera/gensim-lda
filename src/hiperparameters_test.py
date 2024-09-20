### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from utils import *
from preprocess import build_vocab
from metrics import *
import numpy as np
import sys
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("running hyperparameters.py")
        sys.exit(1)

    create_folder_if_not_exists(Path.result.value)
    create_folder_if_not_exists(Path.result_default_hiper.value)
    create_folder_if_not_exists(Path.result_fixed_hiper.value)
    create_folder_if_not_exists(Path.result_word_cloud.value)
    
    config_path = sys.argv[1]
    configuracion = load_config(config_path)
        
    left_topics = configuracion['lk_hyperparameters']
    right_topics = configuracion['rg_hyperparameters']
    lalpha = configuracion['lalpha_hyperparameters']
    ralpha = configuracion['ralpha_hyperparameters']
    alpha_step = configuracion['alpha_step_hyperparameters']
    lbeta = configuracion['lbeta_hyperparameters']
    rbeta = configuracion['rbeta_hyperparameters']
    beta_step = configuracion['beta_step_hyperparameters']
    
    _alpha = list(np.arange(lalpha, ralpha, alpha_step))
    _alpha.append('symmetric')
    _alpha.append('asymmetric')

    _beta = list(np.arange(lbeta, rbeta, beta_step))
    _beta.append('symmetric')
    
    texts = load_file(Path.dataset_path.value)
    tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    
    # tokenized_texts = joblib.load('data/tokenized_texts.joblib')
    # _dict = joblib.load('data/_dict.joblib')
    # trans_TFIDF = joblib.load('data/trans_TFIDF.joblib')
    # ldaModel = joblib.load('data/ldaModel.joblib')
    # ldaModel = lda(trans_TFIDF, num_topics =3, id2word =_dict)                
    # index = 0 
    
    print('Builded vocabulary ✅')
    
    coherences = []
    for k in range(left_topics, right_topics + 1):
        for alpha in _alpha:
            for beta in _beta:
                ldaModel = lda(trans_TFIDF, num_topics =k, id2word =_dict, alpha= alpha, eta= beta)                
                cohe = coherence(ldaModel, tokenized_texts, _dict)
                
                # cohe = index
                
                print(f'K:{k}  Alpha: {alpha}  Beta: {beta} Coherence: {cohe}')
                
                result = {
                    'k': k,
                    # 'k': 3,
                    'alpha': alpha,
                    'beta': beta,
                    'coherence': cohe,
                    'model': ldaModel
                }
                coherences.append(result)
                
                # index += 1
                
    models = sorted(coherences, key= lambda _dict: _dict['coherence'], reverse= True)
    
    results = [{
                'k': item['k'],
                'alpha': item['alpha'],
                'beta': item['beta'],
                'coherence': item['coherence']
                } for item in models]
    
    best_model = models[0]  
    topic_word_cloud(best_model['model'], best_model['k'], save=True, path= Path.result_word_cloud.value)
    
    # joblib.dump(results, f'{Path.result_fixed_hiper.value}results.joblib')
    json.dump(results, open(f'{Path.result_fixed_hiper.value}results.json', 'w'))
    json.dump({'k': best_model['k'], 'alpha': best_model['alpha'], 'beta': best_model['beta']}, open(f'{Path.result_fixed_hiper.value}fixed_hyperparameters.json', 'w'))
    save_result_csv(best_model['model'], best_model['k'], Path.result_topics_hiper.value) 
    
    k, alpha, beta, cohe = best_model['k'], best_model['alpha'], best_model['beta'], best_model['coherence']
    print('-'*25, '✨ Best Model' , '-'*25)
    print(f'K: {k}  Alpha: {alpha}  Beta: {beta} Coherence: {cohe}')
    print('-'*50)