import numpy as np
import logging
from utils import save_agent, save_results, record_agent, eval_agent
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
writer = SummaryWriter()

def train_agent(agent, env, game_name: str, num_episodes: int, new_agent: bool, episode_score: list, eval_episode_score: dict, num_frames: int, save_freq: int, record_freq: int, eval_freq: int, num_eval_runs: int, update_freq_target: int, enable_starter_action: bool, starter_action: int, update_freq_policy: int = 4):
    """
    Trains the agent on the environment over a specified number of episodes.
    """
    
    if new_agent:
        
        logger.info("Agent's memory has been initialized")
        
        # Update agent's memory with random actions until full (i.e. agent memory index is zero again)
        state, info = env.reset()
        agent.cache(state, new_episode = True)
        done = False
        num_lives = info['lives']
        life_lost = False 
        new_game = True
        
        while(agent.memory.ind != 0):
            
            if done:
                state, info = env.reset()
                agent.cache(state, new_episode = True)
                done = False
                num_lives = info['lives']
                life_lost = False 
                new_game = True
                
            if enable_starter_action and (new_game or life_lost):
                action = starter_action
                new_game = False
                life_lost = False  
            
            else:
                action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.cache(next_state, action, reward, done)
            
            # Check for life loss condition
            if info['lives'] < num_lives:
                life_lost = True
                num_lives = info['lives']
                    
            
            if ((agent.memory.ind + 1) % 10000) == 0:
                progress = (agent.memory.ind + 1) / agent.memory.buffer_size * 100
                logger.info(f"Progress: {int(progress)}%")
            
        save_agent(agent= agent, game = game_name, new_game = True)
        
        logger.info("Agent's memory is now full with initial data.")
        
        # Display previous info on Tensorboard

    starting_episode = len(episode_score)
    # Start training agent
    for i_episode in range(num_episodes):

        # Intialise environment
        steps_since_update = 0
        state, info = env.reset()
        agent.cache(state, new_episode = True)
        total_reward = 0
        done = False
        num_lives = info['lives']
        life_lost = False 
        new_game = True

        while not done:
            
            # Force the first action to be 1 (FIRE) if new game or life lost to speed up game
            if enable_starter_action and (new_game or life_lost):
                action = starter_action
                new_game = False
                life_lost = False  
            else:
                action = agent.behaviour_policy(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Increment frame count
            num_frames += 1
            
            # Store information
            done = terminated or truncated
            agent.cache(next_state, action, reward, done)

            # Update reward
            total_reward += reward
            
            # Update state
            state = next_state
                    
            # Updating policy and target (using soft update) networks. Update occurs on fourth action of agent by default.
            steps_since_update += 1
            if steps_since_update >= update_freq_policy:
                current_loss = agent.update_policy()
                steps_since_update = 0
                
                if num_frames % update_freq_target == 0:
                    agent.update_target_hard()
            
            # Check for life loss condition
            if info['lives'] < num_lives:
                life_lost = True
                num_lives = info['lives']
                    
        episode_score.append(total_reward)
        
        # Evaluate agent
        if (starting_episode + i_episode + 1) % eval_freq == 0:
            print("evaluating")
            avg_score = eval_agent(
                agent = agent, 
                env = env, 
                num_runs = num_eval_runs,
                enable_starter_action = enable_starter_action,
                starter_action = starter_action
            )
            eval_episode_score["num_episode"].append(starting_episode + i_episode + 1)
            eval_episode_score["score"].append(avg_score)
            writer.add_scalar("Evaluation", avg_score, starting_episode + i_episode + 1)
        
        # Log some information
        writer.add_scalar("Return", total_reward, starting_episode + i_episode + 1)
        writer.add_scalar("Episilon", agent.eps_threshold, num_frames) 
        writer.add_scalar("Loss", current_loss, num_frames)     
        total_grad_l2_norm = 0
        for _, (name, weight_or_bias_parameters) in enumerate(agent.policy_net.named_parameters()):
            grad_l2_norm = weight_or_bias_parameters.grad.data.norm(p=2).item()
            writer.add_scalar(f'grad_norms/{name}', grad_l2_norm, num_frames)
            total_grad_l2_norm += grad_l2_norm ** 2
        total_grad_l2_norm = total_grad_l2_norm ** (1/2)
        writer.add_scalar('grad_norms/total', grad_l2_norm, num_frames)
        
        # Print progress
        if ((i_episode + 1) % 100) == 0:
            progress = (i_episode + 1) / num_episodes * 100
            logger.info(f"Progress: {int(progress)}%")
            logger.info(f"Number of frames: {num_frames}")
            
        # Save agent and episode scores
        if (i_episode + 1) % save_freq == 0:
            save_agent(
                agent = agent, 
                game = game_name,
                new_game = False,
                episode_score = episode_score,
                eval_episode_score = eval_episode_score,
                num_frames = num_frames,
                num_episode = starting_episode + i_episode + 1
            )
            
        # Record test run of agent and graph of performance
        if (i_episode + 1) % record_freq == 0:
            logger.info("Record agent performance")
            save_results(
                game = game_name, 
                episode_score = episode_score, 
                eval_episode_score = eval_episode_score, 
                num_episode = starting_episode + i_episode + 1
            )
            record_agent(
                agent = agent, 
                env = env, 
                game = game_name, 
                num_episode = starting_episode + i_episode + 1,
                enable_starter_action = enable_starter_action,
                starter_action = starter_action
            )