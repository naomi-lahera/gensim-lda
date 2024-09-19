import ir_datasets
from utils import *

def build():
    dataset = ir_datasets.load("cranfield")
    print('__loaded-dataset__')
    
    texts = []
    for doc in dataset.docs_iter()[:20]:
        texts.append(doc[2])
        print(doc[2][:5])
    
    save_file(texts, Path.dataset)
    
def __init__():
    print('__init__')
    build()