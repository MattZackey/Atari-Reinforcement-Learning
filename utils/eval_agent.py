import numpy as np 
    
def eval_agent(agent, env, num_runs, enable_starter_action, starter_action): 
    """
    Evaluates the agent over num_runs episodes with an epsilon greedy strategy of 0.05 
    """
    
    eval_episodes_scores = []
    
    for _ in range(num_runs):
        
        # Intialise environment
        state, info = env.reset()
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
                sample = np.random.rand()
                
                if sample > 0.05:
                    action = agent.target_policy(state)
                    
                else:    
                    action = np.random.randint(agent.action_dim)
                
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            # Update reward
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Check for life loss condition
            if info['lives'] < num_lives:
                life_lost = True
                num_lives = info['lives']
                    
        eval_episodes_scores.append(total_reward)
    
    avg_score = np.mean(eval_episodes_scores)
    
    return avg_score   