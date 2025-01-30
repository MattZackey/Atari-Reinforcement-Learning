import logging
import torch
import numpy as np
import random
import sys
import argparse
from agents import AgentDQN
from training import train_agent
from envs import setup_atari_env
from utils import load_agent, setup_logger, load_config   

def main(config, game_name):
    
    if game_name not in config["games"]:
        logging.error(f"Game '{game_name}' not found in the configuration.")
        sys.exit(1)

    game_config = config["games"][game_name]
    
    save_frequency = game_config["save_frequency"]
    # if save_frequency < 500:
    #     logging.error("Save frequency must be at least 500.")
    #     sys.exit(1)  
    
    # Setup environment
    env, action_dim, frame_height, frame_width = setup_atari_env(
        game_name = game_name, 
        n_frames = game_config["n_frames"]
    )
    
    # Initialize an agent
    if game_config["new_agent"]:
        
        # Agent Initialization
        agent = AgentDQN(
            action_dim = action_dim,
            n_frames = game_config["n_frames"],
            frame_height = frame_height,
            frame_width = frame_width,
            intial_exploration = game_config["initial_exploration"],
            final_exploration = game_config["final_exploration"],
            final_exploration_frame = game_config["final_exploration_frame"],
            size_memory = game_config["size_memory"],
            batch_size = game_config["batch_size"],
            gamma = game_config["gamma"],
            tau = game_config["tau"],
            learning_rate = game_config["learning_rate"]
        )
        episode_score = []
        num_frames = 0
        logger.info("New Agent Initialized")
        
        logger.info("--------Configuration details--------")
        for key, value in game_config.items():
            logger.info(f"   {key}: {value}")
        logger.info("-------------------------------------")
        
    # Load an existing agent
    else:
        # Load an existing agent
        agent, episode_score, num_frames = load_agent(
            agent_path = game_config["agent_load_path"], 
            game_info_path = game_config["game_info_load_path"]
        )
        
        logger.info(f"Loaded agent from {game_config['agent_load_path']}.")
        logger.info(f"Agent has already been trained for {len(episode_score)} episodes and {num_frames} frames.")

    # Train the agent
    logger.info("Starting training...")
    train_agent(
        agent = agent, 
        env = env, 
        game_name = game_name,
        num_episodes = game_config["num_episodes"], 
        new_agent = game_config["new_agent"], 
        episode_score = episode_score,
        num_frames = num_frames,
        save_freq = game_config["save_frequency"], 
        record_freq = game_config["record_frequency"],
        update_freq_target = game_config["update_freq_target"]
    )
    logger.info("Training completed.")

if __name__ == "__main__":
    # Argument parser for dynamic game selection
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari games.")
    parser.add_argument(
        "--game",
        type=str,
        required=True
    )
    args = parser.parse_args()

    # Load configuration from the current directory
    config = load_config('config.yaml')

    # Initialize logger
    setup_logger("results/training.log")
    logger = logging.getLogger(__name__)

    # Set random seeds for reproducibility
    seed_value = config["games"][args.game]["seed_value"]
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Run the main function with the selected game
    main(config, args.game)
