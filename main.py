import logging
import torch
import numpy as np
import random
import sys
import argparse
from agents import AgentDQN
from training import train_agent
from envs import setup_atari_env
from utils import load_agent, setup_logger, check_s3_bucket, create_s3_keys, str_to_bool, create_results_directories

def main(args):
    
    # Convert to bool
    new_agent = str_to_bool(args.new_agent)
    train_on_aws = str_to_bool(args.train_on_aws)
    enable_starter_action = str_to_bool(args.enable_starter_action)
    
    if args.save_frequency < 1000:
        
        logging.error("Save frequency must be at least 1000.")
        sys.exit(1)  
        
    if train_on_aws: # Setup on S3
        
        logger.info("Setting up S3")
        
        check_s3_bucket(
            bucket_name = args.s3_bucket_name
            )
        
        create_s3_keys(
            bucket_name = args.s3_bucket_name, 
            save_root_folder = args.save_root_folder, 
            game_name = args.game
        )
    
    else: # Setup local
        
        create_results_directories(
            save_root_folder = args.save_root_folder,
            game_name = args.game
        )
        
    # Setup environment
    env, action_dim, frame_height, frame_width = setup_atari_env(
        game_name = args.game, 
        n_frames = args.n_frames
    )
    
    # Initialize an agent
    if args.new_agent:
        
        # Agent Initialization
        agent = AgentDQN(
            action_dim = action_dim,
            n_frames = args.n_frames,
            frame_height = frame_height,
            frame_width = frame_width,
            intial_exploration = args.initial_exploration,
            final_exploration = args.final_exploration,
            final_exploration_frame = args.final_exploration_frame,
            size_memory = args.size_memory,
            batch_size = args.batch_size,
            gamma = args.gamma,
            tau = args.tau,
            learning_rate = args.learning_rate
        )
        episode_score = []
        eval_episode_score = {"num_episode": [], "score": []}
        num_frames = 0
        logger.info("New Agent Initialized")
        
        logger.info("--------Configuration details--------")
        for key, value in vars(args).items():
            logger.info(f"   {key}: {value}")
        logger.info("-------------------------------------")
        
    # Load an existing agent
    else:
        agent, episode_score, eval_episode_score, num_frames = load_agent(
            agent_path = args.agent_load_path, 
            game_info_path = args.game_info_load_path
        )
        
        logger.info(f"Loaded agent from {args.agent_load_path}.")
        logger.info(f"Agent has already been trained for {len(episode_score)} episodes and {num_frames} frames.")

    # Train the agent
    logger.info("Starting training...")
    train_agent(
        agent = agent, 
        env = env, 
        game_name = args.game,
        num_episodes = args.num_episodes, 
        new_agent = new_agent, 
        episode_score = episode_score,
        eval_episode_score = eval_episode_score,
        num_frames = num_frames,
        save_freq = args.save_frequency, 
        record_freq = args.record_frequency,
        save_root_folder = args.save_root_folder,
        eval_freq = args.evaluation_frequency,
        num_eval_runs = args.num_evaluation_runs,
        update_freq_target = args.update_freq_target,
        enable_starter_action = enable_starter_action,
        starter_action = args.starter_action,
        train_on_aws = train_on_aws,
        s3_bucket_name = args.s3_bucket_name
    )
    logger.info("Training completed.")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari games.")
    
    # Agent Hyperparameters
    parser.add_argument("--n_frames", type=int, default=4, help="Number of frames in a state")
    parser.add_argument("--seed_value", type=int, default=42, help="Random seed value")
    parser.add_argument("--initial_exploration", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--final_exploration", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--final_exploration_frame", type=int, default=1000000, help="Frame at which exploration ends")
    parser.add_argument("--size_memory", type=int, default=1000000, help="Size of the memory buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update factor for target networks")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")   
    
    # Game Specific Hyperparameters and Arguments
    parser.add_argument("--game", type=str, required=True, help="The name of the game")
    parser.add_argument("--num_episodes", type=int, default=20000, help="Number of episodes to run")
    parser.add_argument("--new_agent", type=str, default="true", help="Whether to initialize a new agent or load an existing one")
    parser.add_argument("--agent_load_path", type=str, default="", help="Path to load the agent model from")
    parser.add_argument("--game_info_load_path", type=str, default="", help="Path to load the game info from")
    parser.add_argument("--save_frequency", type=int, default=5000, help="Frequency (in episodes) to save the model")
    parser.add_argument("--record_frequency", type=int, default=1000, help="Frequency (in episodes) to record the gameplay")
    parser.add_argument("--evaluation_frequency", type=int, default=50, help="Frequency (in episodes) to run evaluation")
    parser.add_argument("--num_evaluation_runs", type=int, default=20, help="Number of evaluation runs per evaluation cycle")
    parser.add_argument("--update_freq_target", type=int, default=10000, help="Frequency of target network updates if hard update selected")
    parser.add_argument("--enable_starter_action", type=str, default="true", help="Whether to enable the starter action")
    parser.add_argument("--starter_action", type=int, default=1, help="The action to start with if starter action is enabled")
    parser.add_argument("--save_root_folder", type=str, required=True, help="Root folder where all outputs (models, logs, recordings) will be saved")
    
    # AWS parameters
    parser.add_argument("--train_on_aws", type=str, default="false", help="Option to train model on AWS, otherwise defaults to local machine")
    parser.add_argument("--s3_bucket_name", type=str, default="", help="S3 Bucket")
    
    args = parser.parse_args()

    setup_logger("training.log")
    logger = logging.getLogger(__name__)

    seed_value = args.seed_value
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    main(args)