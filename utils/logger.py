import logging

# def setup_logger(log_level, log_file = "training"):
#     """
#     Sets up the logger to log to both a file and the console.
#     """
#     # Create a logger object
#     logger = logging.getLogger(__name__)
#     log_level = getattr(logging, log_level.upper())
#     logger.setLevel(log_level) 

#     if logger.hasHandlers() and logger.has:
#         return logger

#     # Create handlers
#     file_handler = logging.FileHandler(f"results/{log_file}.log")
#     stream_handler = logging.StreamHandler()

#     # Create a formatter and attach it to the handlers
#     formatter = logging.Formatter(" %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     stream_handler.setFormatter(formatter)

#     # Add the handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)

#     return logger



# # Create a logger object
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) 

# # Check if handlers are already added
# if not logger.hasHandlers():
#     # Create handlers
#     file_handler = logging.FileHandler(f"results/training.log")
#     stream_handler = logging.StreamHandler()

#     # Create a formatter and attach it to the handlers
#     formatter = logging.Formatter(" %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     stream_handler.setFormatter(formatter)

#     # Add the handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
    
    
    
def setup_logger(LOG_FILE):
    
    logging.basicConfig(
        level = logging.INFO,
        format = '%(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )