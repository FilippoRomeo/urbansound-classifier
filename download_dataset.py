# download_dataset.py

import soundata
from config import DATASET_PATH

def download_and_validate():
    dataset = soundata.initialize('urbansound8k', data_home=DATASET_PATH)
    print("Downloading UrbanSound8K...")
    dataset.download()
    print("Validating dataset files...")
    dataset.validate()
    print("Dataset is ready!")

if __name__ == "__main__":
    download_and_validate()
