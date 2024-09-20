# gensim-lda

## Latente Dirichlet Allocation (LDA)
LDA es un modelo generativo probabilístico que utiliza distribuciones de Dirichlet para inferir temas a partir de documentos. Utiliza un enfoque bayesiano para analizar los datos, utilizando probabilidades condicionales para inferir temas basados en las palabras presentes en los documentos que analiza. Al terminar su entrenamiento por cada documento LDA habrá calculado la proporción en la que ocurren cada uno de los K tópicos que inicialmente debían ser conformados.  Es necesario aclarar que el número de tópicos a extraer es un hiperparámetro para este modelo. Elegir el número de tópicos en LDA es una decisión importante que influye directamente en los resultados obtenidos.

## Implementación y evaluación de LDA
Utilizando la implementación de la librería Gensim de Python, se verá la influencia de diferentes configuraciones de hiperparámetros en el rendimiento del modelo LDA. Los resultados serán evaluados mediante visualizacion (Word Cloud) y a través de la métrica de Coherencia.

## Ejecución
Solo debe ejecutar los siguientes archivos, ya sea directamente en e l directorio o a través de la consola. 
- El primer paso es cargar le dataset: ``EXE-LOAD-DATASET.bat``
- Entrenamineto del modelo con los parámetros por defecto de gensim: ``EXE-FIT-EVAL.bat``
- Análisis de distintas configuraciones de los hiperparámetros: ``EXE-HYPERPARAMETERS-TEST ``
