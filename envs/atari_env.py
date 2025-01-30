import gymnasium
import ale_py
from ale_py import ALEInterface
ale = ALEInterface()

from gymnasium.wrappers import (
    ResizeObservation,
    FrameStack,
)

def setup_atari_env(game_name: str, n_frames: int):
    """
    Sets up the Atari environment with frame stacking, and returns important environment information.
    """
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(game_name, obs_type="grayscale", render_mode="rgb_array")
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=n_frames)

    frame_height = env.observation_space.shape[1]
    frame_width = env.observation_space.shape[2]
    
    return env, env.action_space.n, frame_height, frame_width