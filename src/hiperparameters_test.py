### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from utils import *
from preprocess import build_vocab
from metrics import *
import numpy as np
import sys
import csv
    
def print_topics(ldaModel):
    pprint(ldaModel.print_topics())
    
def save_result_csv(model, num_topics, num_words):
    print('Saving topics matrix...')
    
    # Obtener los tópicos del modelo (sin formato para extraer palabra y ponderación)
    topics = model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

    # Extraer todas las palabras únicas en los tópicos
    all_words = set()
    for topic_num, topic in topics:
        words = [word for word, _ in topic]
        all_words.update(words)
    all_words = sorted(all_words)  # Ordenar las palabras alfabéticamente

    create_file_if_not_exists(f'{Path.result_fixed_hiper}topics.csv')

    with open(f'{Path.result_fixed_hiper.value}topics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tópico'] + all_words)

        for topic_num, topic in topics:
            word_dict = {word: weight for word, weight in topic}
            row = [word_dict.get(word, 0) for word in all_words]
            writer.writerow([f"Tópico {topic_num}"] + row)
    
    print('Done ✅')
    
    
def test():
    m  = joblib.load('data/ldaModel.joblib')
    coherences = [{
                    'k': 1,
                    'alpha': 2,
                    'beta': 3,
                    'coherence': 4,
                    'model': m
                },
                {
                    'k': 1,
                    'alpha': 2,
                    'beta': 3,
                    'coherence': 5,
                    'model': m
                }]        
       
    models = sorted(coherences, key= lambda _dict: _dict['coherence'], reverse= True)
    print(models)
    
    results = [{
                'k': item['k'],
                'alpha': item['alpha'],
                'beta': item['beta'],
                'coherence': item['coherence']
                } for item in models]
    
    # create_file_if_not_exists(f'{Path.result_fixed_hiper}results.joblib')
    joblib.dump(results, f'{Path.result_fixed_hiper.value}results.joblib')
    
    best_model = models[0]
        
    topic_word_cloud(best_model['model'], best_model['k'], save=True, path= Path.result_word_cloud.value)
    # print('Coherence Score:', best_model['coherence'])
    
    # create_file_if_not_exists(f'{Path.result_fixed_hiper}fixed_hyperparameters.json')
    json.dump({'k': best_model['k'], 'alpha': best_model['alpha'], 'beta': best_model['beta']}, open(f'{Path.result_fixed_hiper.value}fixed_hyperparameters.json', 'w'))
    
    _dict = joblib.load('data/_dict.joblib')
    save_result_csv(best_model['model'], best_model['k'], _dict.num_pos)
    
    print('-'*25, '✨Best Model' , '-'*25)
    print(f'K:{best_model['k']}  Alpha: {best_model['alpha']}  Beta: {best_model['beta']} Coherence: {best_model['coherence']}')
    print('-'*50)

        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("running hyperparameters.py")
        sys.exit(1)

    create_folder_if_not_exists(Path.result.value)
    create_folder_if_not_exists(Path.result_default_hiper.value)
    create_folder_if_not_exists(Path.result_fixed_hiper.value)
    create_folder_if_not_exists(Path.word_cloud.value)
    
    config_path = sys.argv[1]
    configuracion = load_config(config_path)
    
    num_topics = configuracion['k_fit_eval']
    
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
    
    texts = load_file(Path.dataset_path)
    # tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    
    tokenized_texts = joblib.load('data/tokenized_texts.joblib')
    _dict = joblib.load('data/_dict.joblib')
    trans_TFIDF = joblib.load('data/trans_TFIDF.joblib')
    
    print('Builded vocabulary ✅')
    
    coherences = []
    for k in range(left_topics, right_topics + 1):
        for alpha in _alpha:
            for beta in _beta:
                ldaModel = lda(trans_TFIDF, num_topics =k, id2word =_dict, alpha= alpha, eta= beta)                
                cohe = coherence(ldaModel, tokenized_texts, _dict)
                
                print(f'K:{k}  Alpha: {alpha}  Beta: {beta} Coherence: {cohe}')
                
                result = {
                    'k': k,
                    'alpha': alpha,
                    'beta': beta,
                    'coherence': cohe,
                    'model': ldaModel
                }
                coherences.append(result)
                
    models = sorted(coherences, key= lambda _dict: _dict['coherence'], reverse= True)
    print(models)
    
    results = [{
                'k': item['k'],
                'alpha': item['alpha'],
                'beta': item['beta'],
                'coherence': item['coherence']
                } for item in models]
    
    best_model = models[0]  
    topic_word_cloud(best_model['model'], best_model['k'], save=True, path= Path.result_word_cloud.value)
    
    joblib.dump(results, f'{Path.result_fixed_hiper.value}results.joblib')
    json.dump({'k': best_model['k'], 'alpha': best_model['alpha'], 'beta': best_model['beta']}, open(f'{Path.result_fixed_hiper.value}fixed_hyperparameters.json', 'w'))
    save_result_csv(best_model['model'], best_model['k'], _dict.num_pos)
    
    print('-'*25, '✨ Best Model' , '-'*25)
    print(f'K:{best_model['k']}  Alpha: {best_model['alpha']}  Beta: {best_model['beta']} Coherence: {best_model['coherence']}')
    print('-'*50)