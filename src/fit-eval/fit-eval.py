### ------------------------------------------------------------- TRAINIG LDA --------------------------------------------------------------------- ###
### ------------------------------------------------------------- USING gensim --------------------------------------------------------------------- ###
from gensim.models import LdaModel as lda
from pprint import pprint
from src.utils import *
from src.preprocess import build_vocab
from src.metrics import *
    
def print_topics(ldaModel):
    pprint(ldaModel.print_topics())
        
if __name__ == '__main__':
    num_topics = 3
    texts = load_file(Path.dataset_path)
    tokenized_texts, _dict, trans_TFIDF = build_vocab(texts)
    print('Builded vocabulary ✅')
    
    ldaModel = lda(trans_TFIDF, num_topics =num_topics, id2word =_dict)
    save_file(ldaModel, Path.lda_model_path)
    print('Trained model ✅')
    
    topic_word_cloud(ldaModel)
    
    coherence_lda = coherence(ldaModel, tokenized_texts, _dict)
    print('\nCoherence Score:', coherence_lda)
    print('Evaluated model ✅')
    