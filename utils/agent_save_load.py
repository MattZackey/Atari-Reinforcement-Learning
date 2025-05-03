import numpy as np
import os
import pickle
import logging
import matplotlib.pyplot as plt
import yaml
import boto3

logger = logging.getLogger(__name__)

def save_agent(agent, game, new_game, save_root_folder, train_on_aws ,s3_bucket_name, episode_score = [], eval_episode_score = {}, num_frames = 0, num_episode = 0):
    """
    Saves agent and its scores
    """
    
    if new_game:
        game_info = {"scores": [], "eval_scores": {"num_episode": [], "score": []} ,"num_frames": 0}
    else: 
        game_info = {"scores": episode_score, "eval_scores": eval_episode_score,"num_frames": num_frames}
                
    
    if train_on_aws:
        s3 = boto3.client('s3')
        
        # Saving agent
        with open(f"/opt/ml/output/{game}/agent/agent.pkl", "wb") as f:
            pickle.dump(agent, f)
            
        # Save agent's resuls
        with open(f"/opt/ml/output/{game}/game/game_info.pkl", "wb") as f:
            pickle.dump(game_info, f)
        
        # Upload files to S3
        s3.upload_file(Filename = f"/opt/ml/output/{game}/agent/agent.pkl", Bucket = s3_bucket_name, Key = f"{save_root_folder}/{game}/agent/agent_episode_{num_episode}.pkl")
        s3.upload_file(Filename = f"/opt/ml/output/{game}/game/game_info.pkl", Bucket = s3_bucket_name, Key = f"{save_root_folder}/{game}/game/game_info_{num_episode}.pkl")
    
    else:    
        # Saving results locally

        # Save agent
        with open(f"{save_root_folder}/{game}/agent/agent_episode_{num_episode}.pkl", "wb") as f:
            pickle.dump(agent, f)
            
        # Save agent's results
        with open(f"{save_root_folder}/{game}/game/game_info_{num_episode}.pkl", "wb") as f:
            pickle.dump(game_info, f)


def save_results(game, episode_score, eval_episode_score, num_episode, save_root_folder, train_on_aws, s3_bucket_name):
    """
    Saves graph of agent's performance
    """
            
    # Plot episode scores for training
    plt.plot(np.arange(len(episode_score)), episode_score, linewidth=0.2)
    plt.title(f"Training Scores - {game} - Episode {num_episode}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    if train_on_aws:
        plt.savefig(f"/opt/ml/output/{game}/plots/episode_scores.png")
    else:
        plt.savefig(f"{save_root_folder}/{game}/plots/episode_scores_{num_episode}.png")
    plt.close()  
    
    # Plot episode scores for evaluation
    plt.plot(eval_episode_score["num_episode"], eval_episode_score["score"], linewidth=0.2)
    plt.title(f"Evaluation Scores - {game} - Episode {num_episode}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    if train_on_aws:
        plt.savefig(f"/opt/ml/output/{game}/plots/eval_episode_scores.png")
    else:
        plt.savefig(f"{save_root_folder}/{game}/plots/eval_episode_scores_{num_episode}.png")
    plt.close()  
        
    # Upload plots to S3
    if train_on_aws:
        s3 = boto3.client('s3')
        s3.upload_file(Filename = f"/opt/ml/output/{game}/plots/episode_scores.png", Bucket = s3_bucket_name, Key = f"{save_root_folder}/{game}/plots/episode_scores_{num_episode}.png")
        s3.upload_file(Filename = f"/opt/ml/output/{game}/plots/eval_episode_scores.png", Bucket = s3_bucket_name, Key = f"{save_root_folder}/{game}/plots/eval_episode_scores_{num_episode}.png")
   
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