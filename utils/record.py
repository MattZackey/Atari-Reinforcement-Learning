import imageio
import os
import boto3
    
def record_agent(agent, env, game, num_episode, enable_starter_action, starter_action, save_root_folder, train_on_aws ,s3_bucket_name): 
    """
    Records a test run of the agent in a given environment and saves it as a GIF.
    """
    
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
    
    if train_on_aws:
        imageio.mimsave(f"/opt/ml/output/{game}/gameplay/agent.gif", frames, fps=30)
        
        # Write to ss
        s3 = boto3.client('s3')
        s3.upload_file(Filename = f"/opt/ml/output/{game}/gameplay/agent.gif", Bucket = s3_bucket_name, Key = f"{save_root_folder}/{game}/gameplay/agent_{num_episode}.gif")
    
    else:
        imageio.mimsave(f"{save_root_folder}/{game}/gameplay/agent_{num_episode}.gif", frames, fps=30)