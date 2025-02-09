import logging

def setup_logger(LOG_FILE):
    
    logging.basicConfig(
        level = logging.INFO,
        format = '%(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )