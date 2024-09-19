import joblib
from enum import Enum
import json
import sys
import os

class Path(Enum):
    dataset_path = 'data/dataset.joblib'
    nlp_path = 'data/nlp_model.joblib'
    lda_model_path = 'data/ldaModel.joblib'
    different_alphas = 'data/different_alphas.joblib'
    dataset = 'data/dataset.joblib'
    data = 'data/'
    
def load_config(config_path):
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
        
def create_file_if_not_exists(file_path):
    # Obtener el directorio del archivo
    directory = os.path.dirname(file_path)  
    # Crear el directorio si no existe
    os.makedirs(directory, exist_ok=True)   
    # Crear el archivo si no existe
    open(file_path, 'a').close()
    
def create_folder_if_not_exists(path):
    try:
        os.makedirs(path)
        print(f"✅ Created foler: {path}")
    except FileExistsError:
        print(f"Folder {path} already exists.")
    except OSError as error:
        print(f"Error creating folder: {error}")

def load_file(path: Path):
    try: 
        return joblib.load(path.value)
    except Exception as e :
        Exception(f'❌ Error loading file. {e}')

def save_file(data, path: Path):
    try:
        joblib.dump(data, path.value)
    except FileNotFoundError:
        create_file_if_not_exists(Path.dataset_path)
        joblib.dump(data, path.value)
    except Exception as e:
        Exception(f'❌ Error saving file. {e}')