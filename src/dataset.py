import ir_datasets
from utils import *

def build():
    dataset = ir_datasets.load("cranfield")
    
    texts = []
    for doc in dataset.docs_iter()[:700]:
        texts.append(doc[2])
    
    save_file(texts, Path.dataset.value)
    
if __name__ == '__main__':    
    create_folder_if_not_exists(Path.data.value)
    
    print('Loading...')
    build()
    print('✅ Loaded dataset (700 texts).')
    