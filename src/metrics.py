from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import *
import os

def topic_word_cloud(ldaModel, num_topics, topic_words_freqs = None, save= False, path=''):
    if save:
        create_folder_if_not_exists(path)
    
    topic = 0 # Initialize counter
    while topic < num_topics:
        # Get topics and frequencies and store in a dictionary structure
        if not topic_words_freqs: topic_words_freq = dict(ldaModel.show_topic(topic, topn=50)) # NB. the 'dict()' constructor builds dictionaries from sequences (lists) of key-value pairs - this is needed as input for the 'generate_from_frequencies' word cloud function
        else: topic_words_freq = topic_words_freqs[topic]
        
        # Generate Word Cloud for topic using frequencies
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        if save:
            # Guardar la imagen en el path especificado
            file_path = os.path.join(path, f"topic_{topic}.png")
            plt.savefig(file_path)  # Guardar el archivo en formato PNG
            plt.close()  # Cerrar la figura para evitar sobreescribir
        
        topic += 1
        
def coherence(ldaModel, tokenized_texts, _dict):    
    coherence_model_lda = CoherenceModel(model=ldaModel, texts=tokenized_texts, dictionary=_dict, coherence='c_v')
    print('Modelo de Coherencia Creado')
    
    print('Calculating coherence score...')
    return coherence_model_lda.get_coherence()