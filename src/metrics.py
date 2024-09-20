from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import *
import os

def topic_word_cloud(ldaModel, num_topics, save= False, path=''):
    if save:
        create_folder_if_not_exists(path)
    
    topic = 0 
    while topic < num_topics:
        # Get topics and frequencies and store in a dictionary structure
        topic_words_freq = dict(ldaModel.show_topic(topic, topn=50)) 
        
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        # plt.show()
        
        if save:
            file_path = os.path.join(path, f"topic_{topic}.png")
            plt.savefig(file_path, bbox_inches='tight')  # bbox_inches evita espacios extra en blanco
            plt.close() 
            
        topic += 1
        
def coherence(ldaModel, tokenized_texts, _dict):    
    coherence_model_lda = CoherenceModel(model=ldaModel, texts=tokenized_texts, dictionary=_dict, coherence='c_v')
    print('Modelo de Coherencia Creado')
    
    print('Calculating coherence score...')
    return coherence_model_lda.get_coherence()