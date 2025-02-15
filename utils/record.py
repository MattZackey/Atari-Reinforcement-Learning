import imageio
import os
    
def record_agent(agent, env, game, num_episode, enable_starter_action, starter_action): 
    """
    Records a test run of the agent in a given environment and saves it as a GIF.
    """
    
    # Create directory
    if not os.path.exists(f"results/{game}/gameplay"):
        os.makedirs(f"results/{game}/gameplay/")
    
    frames = []
    state, info = env.reset() 
    done = False
    num_lives = info['lives']
    life_lost = False 
    new_game = True

    while not done:
        frames.append(env.render())

        if enable_starter_action and (new_game or life_lost):
            action = starter_action  # Force the first action to be 1 after life loss to ensure no infinite loop
            new_game = False
            life_lost = False  
        else:
            action = agent.target_policy(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        done = terminated or truncated
        
        # Check for life loss condition
        if info['lives'] < num_lives:
            life_lost = True
            num_lives = info['lives']

    env.close()
    imageio.mimsave(f"results/{game}/gameplay/agent_{num_episode}.gif", frames, fps=30)