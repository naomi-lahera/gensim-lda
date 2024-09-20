### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from utils import *
from preprocess import build_vocab
from metrics import *
import sys
    
def print_topics(ldaModel):
    pprint(ldaModel.print_topics())
        
def test(num_topics):
    print('Preprocessing corpus..')
    tokenized_texts = joblib.load('data/tokenized_texts.joblib')
    _dict = joblib.load('data/_dict.joblib')
    trans_TFIDF = joblib.load('data/trans_TFIDF.joblib')
    print('Builded vocabulary ✅')
    
    print('Training model...')
    # ldaModel = load_file(Path.lda_model_path)
    ldaModel = lda(trans_TFIDF, num_topics =num_topics, id2word =_dict, alpha= 'asymmetric', eta= 0.7000000000000001 )
    print('Trained model ✅')

    print('Creating word cloud...')
    topic_word_cloud(ldaModel, num_topics)
    
    print('calculating coherence...')
    coherence_lda = coherence(ldaModel, tokenized_texts, _dict)
    print('\nCoherence Score:', coherence_lda)
    print('Evaluated model ✅')    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python mi_script.py <ruta_config>")
        sys.exit(1)

    config_path = sys.argv[1]
    configuracion = load_config(config_path)
    num_topics = configuracion['k_fit_eval']
    print(f'Number of topics to gnerate {num_topics}')
    
    # test(5)
    
    # print('Preprocessing corpus..')
    # texts = load_file(Path.dataset_path.value)
    # tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    # print('Builded vocabulary ✅')
    
    # print('Training model...')
    # ldaModel = lda(trans_TFIDF, num_topics =num_topics, id2word =_dict)
    # save_file(ldaModel, Path.lda_model_path.value)
    # print('Trained model ✅')

    # print('Creating word cloud...')
    # topic_word_cloud(ldaModel, num_topics)
    
    # print('calculating coherence...')
    # coherence_lda = coherence(ldaModel, tokenized_texts, _dict)
    # print('Coherence Score:', coherence_lda)
    
    # print('Evaluated model ✅')
    
