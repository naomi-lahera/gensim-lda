from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def topic_word_cloud(ldaModel, num_topics, topic_words_freqs = None):
    topic = 0 # Initialize counter
    while topic < num_topics:
        # Get topics and frequencies and store in a dictionary structure
        if not topic_words_freq: topic_words_freq = dict(ldaModel.show_topic(topic, topn=50)) # NB. the 'dict()' constructor builds dictionaries from sequences (lists) of key-value pairs - this is needed as input for the 'generate_from_frequencies' word cloud function
        else: topic_words_freq = topic_words_freqs[topic]
        topic += 1
        # Generate Word Cloud for topic using frequencies
        wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
def coherence(ldaModel, tokenized_texts, _dict):
    coherence_model_lda = CoherenceModel(model=ldaModel, texts=tokenized_texts, dictionary=_dict, coherence='c_v')
    return coherence_model_lda.get_coherence()