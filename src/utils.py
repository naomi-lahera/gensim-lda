import joblib
from enum import Enum

class Path(Enum):
    dataset_path = 'src/data/dataset.joblib'
    nlp_path = 'src/data/nlp_model.joblib'
    lda_model_path = 'src/data/ldaModel.joblib'
    different_alphas = 'src/data/different_alphas.joblib'
    dataset = 'src/data/dataset.joblib'

def load_file(path: Path):
    try: 
        return joblib.load(path.value)
    except Exception as e :
        Exception(f'❌ Error loading file. {e}')

def save_file(data, path: Path):
    assert joblib.dump(data, path.value), '❌ Error saving file'