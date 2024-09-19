import joblib
from enum import Enum
import json
import sys

class Path(Enum):
    dataset_path = 'src/data/dataset.joblib'
    nlp_path = 'src/data/nlp_model.joblib'
    lda_model_path = 'src/data/ldaModel.joblib'
    different_alphas = 'src/data/different_alphas.joblib'
    dataset = 'src/data/dataset.joblib'
    
def cargar_configuracion(config_path):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"✅ Archivo de configuración no encontrado: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("'❌ Error al leer el archivo de configuración. Asegúrate de que es un JSON válido.")
        sys.exit(1)

def load_file(path: Path):
    try: 
        return joblib.load(path.value)
    except Exception as e :
        Exception(f'❌ Error loading file. {e}')

def save_file(data, path: Path):
    assert joblib.dump(data, path.value), '❌ Error saving file'