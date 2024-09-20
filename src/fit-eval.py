### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from utils import *
from preprocess import build_vocab
from metrics import *
import sys
          
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python mi_script.py <ruta_config>")
        sys.exit(1)

    config_path = sys.argv[1]
    configuracion = load_config(config_path)
    print(f'Number of topics to gnerate {100}')
    
    print('Preprocessing corpus..')
    texts = load_file(Path.dataset_path.value)
    tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    print('Builded vocabulary ✅')
    
    print('Training model...')
    ldaModel = lda(trans_TFIDF, id2word =_dict)
    save_file(ldaModel, Path.lda_model_path.value)
    print('Trained model ✅')

    # ldaModel = joblib.load('data/ldaModel.joblib')
    # _dict = joblib.load('data/_dict.joblib')
    # tokenized_texts = joblib.load('data/tokenized_texts.joblib')

    print('Creating word cloud...')
    topic_word_cloud(ldaModel, 100, save= True, path= Path.result_word_cloud_default.value)
    
    print('calculating coherence...')
    coherence_lda = coherence(ldaModel, tokenized_texts, _dict)
    print('Coherence Score:', coherence_lda)
    
    json.dump({'k': 100, 'alpha': 'symetric', 'beta': None, 'coherence': coherence_lda}, open(f'{Path.result_default_hiper.value}fixed_hyperparameters.json', 'w'))
    save_result_csv(ldaModel, 100, Path.result_topics_hiper_default.value) 
    
    print('Evaluated model ✅')
    
