#data/data_processing.py

import logging
from app.recommender import generate_recommendations

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_processing')
file_handler = logging.FileHandler('logs/process.log')
error_handler = logging.FileHandler('logs/errors.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(error_handler)

def process_new_data():
    try:
        logger.info('Rozpoczęcie przetwarzania danych.')
        generate_recommendations()
        logger.info('Przetwarzanie danych zakończone pomyślnie.')
    except Exception as e:
        logger.error(f'Błąd podczas przetwarzania danych: {e}')

if __name__ == "__main__":
    process_new_data()
