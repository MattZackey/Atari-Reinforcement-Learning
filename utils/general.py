import logging
import os

logger = logging.getLogger(__name__)
    
def str_to_bool(value):
    
    value = value.lower()
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        logger.error(f"Invalid input: Expected 'true' or 'false', but received '{value}'")
        raise ValueError("Invalid input: Expected 'true' or 'false'")
    
def create_results_directories(save_root_folder, game_name):
    
    if not os.path.exists(f"{save_root_folder}/{game_name}/"):
        os.makedirs(f"{save_root_folder}/{game_name}/agent/")
        os.makedirs(f"{save_root_folder}/{game_name}/game/")
        os.makedirs(f"{save_root_folder}/{game_name}/gameplay/")
        os.makedirs(f"{save_root_folder}/{game_name }/plots/")
    