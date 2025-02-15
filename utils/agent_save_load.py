import numpy as np
import os
import pickle
import logging
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)

def save_agent(agent, game, new_game, episode_score = [], eval_episode_score = {}, num_frames = 0, num_episode = 0):
    """
    Saves agent and its scores
    """
    
    # Create directories 
    if not os.path.exists(f"results/{game}/"):
        os.makedirs(f"results/{game}/agent/")
        os.makedirs(f"results/{game}/game/")
    
    # Save agent
    with open(f"results/{game}/agent/agent_episode_{num_episode}.pkl", "wb") as f:
        pickle.dump(agent, f)
    
    if new_game:
        game_info = {"scores": [], "eval_scores": {"num_episode": [], "score": []} ,"num_frames": 0}
    
    else: 
        game_info = {"scores": episode_score, "eval_scores": eval_episode_score,"num_frames": num_frames}
             
    # Save agent's results
    with open(f"results/{game}/game/game_info_{num_episode}.pkl", "wb") as f:
        pickle.dump(game_info, f)


def save_results(game, episode_score, eval_episode_score, num_episode):
    
    """
    Saves graph of agent's performance
    """
    
    # Create directory
    if not os.path.exists(f"results/{game}/plots"):
        os.makedirs(f"results/{game}/plots/")
            
    # Plot episode scores for training
    plt.plot(np.arange(len(episode_score)), episode_score, linewidth=0.2)
    plt.title(f"Training Scores - {game} - Episode {num_episode}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig(f"results/{game}/plots/episode_scores_{num_episode}.png")
    plt.close()  
    
    # Plot episode scores for evaluation
    plt.plot(eval_episode_score["num_episode"], eval_episode_score["score"], linewidth=0.2)
    plt.title(f"Evaluation Scores - {game} - Episode {num_episode}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig(f"results/{game}/plots/eval_episode_scores_{num_episode}.png")
    plt.close()  
        
   
def load_agent(agent_path: str, game_info_path: str):
    """
    Loads an existing RL agent from the specified save path with its training scores.
    """
    
    # Load agent
    try:
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
    except FileNotFoundError as fnfe:
        logger.error(f"Agent save path {agent_path} does not exist!")
        raise FileNotFoundError(f"Agent save path {agent_path} does not exist!") from fnfe
    
    # Load agent details
    try:
        with open(game_info_path, "rb") as f:
            game_info = pickle.load(f)
            episode_score = game_info["scores"]
            eval_episode_score = game_info["eval_scores"]
            num_frames = game_info["num_frames"]
            
    except FileNotFoundError as fnfe:
        logger.error(f"Episode score path {game_info_path} does not exist!")
        raise FileNotFoundError(f"Episode score path {game_info_path} does not exist!") from fnfe
        
    return agent, episode_score, eval_episode_score, num_frames

def load_config(config_path):
    """
    Load configuration from a YAML file
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config