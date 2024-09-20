import csv
import joblib

def guardar_topicos_csv():
    lda_model = joblib.load('data/ldaModel.joblib')
    num_topics = 3
    filepath = 'csv_test.csv'
    
    # Abrir el archivo CSV para escritura
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir el encabezado
        header = ['Topic'] + [f'Word_{i}' for i in range(1, 51)]  # 50 palabras por cada tópico
        writer.writerow(header)
        
        # Iterar sobre los tópicos y escribir en el CSV
        for topic_id in range(num_topics):
            topic = lda_model.show_topic(topic_id, topn=50)  # Obtener las 50 palabras más representativas del tópico
            row = [f'Topic {topic_id}'] + [f'{word}: {weight:.4f}' for word, weight in topic]
            writer.writerow(row)

guardar_topicos_csv()
